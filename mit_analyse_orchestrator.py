# -*- coding: utf-8 -*-
# Filename: orchestrator.py
"""
NeuroPersona Workflow Orchestrator

Dieses Skript steuert den 3-stufigen Workflow:
1. Erzeugt strukturierte Daten (CSV/DataFrame) aus einem User-Prompt
   (nutzt aktuell gemini_perception_unit Simulation).
2. Führt die NeuroPersona-Simulation mit diesen Daten durch
   (nutzt neuropersona_core.run_neuropersona_simulation).
3. Generiert eine finale, angereicherte Antwort mithilfe der Gemini API,
   basierend auf dem ursprünglichen Prompt und den NeuroPersona-Ergebnissen.
"""

import os
import time
import pandas as pd
from typing import Callable, Optional, Tuple, Dict, Any
import json # Import für structured_results Anzeige

# --- Konstanten (aus neuropersona_core importiert) ---
# Werden unten im Import-Block gesetzt

# --- Konfiguration ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
FINAL_RESPONSE_MODEL_NAME = 'gemini-2.0-flash'
# CSV_GENERATION_MODEL_NAME = 'gemini-1.5-flash-latest' # Optional für echte CSV-Gen

# --- Modulimporte ---
try:
    import google.generativeai as genai
    gemini_api_available = True
    if not GEMINI_API_KEY:
        print("WARNUNG: Google Generative AI SDK ist installiert, aber GEMINI_API_KEY wurde nicht in den Umgebungsvariablen gefunden. Echte API-Aufrufe werden fehlschlagen.")
except ImportError:
    print("WARNUNG: 'google-generativeai' nicht installiert. Echte Gemini API-Aufrufe sind deaktiviert.")
    print("Installieren Sie es mit: pip install google-generativeai")
    gemini_api_available = False

try:
    from gemini_perception_unit import generate_prompt_based_csv
    from neuropersona_core import run_neuropersona_simulation, \
                                  DEFAULT_EPOCHS as NP_DEFAULT_EPOCHS, \
                                  DEFAULT_LEARNING_RATE as NP_DEFAULT_LR, \
                                  DEFAULT_DECAY_RATE as NP_DEFAULT_DR, \
                                  DEFAULT_REWARD_INTERVAL as NP_DEFAULT_RI
    # Setze lokale Defaults basierend auf den importierten Werten
    DEFAULT_EPOCHS = NP_DEFAULT_EPOCHS
    DEFAULT_LEARNING_RATE = NP_DEFAULT_LR
    DEFAULT_DECAY_RATE = NP_DEFAULT_DR
    DEFAULT_REWARD_INTERVAL = NP_DEFAULT_RI

except ImportError as e:
    print(f"FEHLER: Notwendige Skripte (gemini_perception_unit.py, neuropersona_core.py) nicht gefunden oder Importfehler: {e}")
    exit()

# --- Hilfsfunktionen ---
def _default_status_callback(message: str):
    """Standard-Callback, gibt Status auf der Konsole aus."""
    print(f"[Orchestrator Status] {message}")

def configure_gemini_api():
    """Konfiguriert die Gemini API, falls verfügbar und Key vorhanden."""
    if gemini_api_available and GEMINI_API_KEY:
        try:
            genai.configure(api_key=GEMINI_API_KEY)
            print("Gemini API konfiguriert.")
            return True
        except Exception as e:
            print(f"FEHLER bei der Konfiguration der Gemini API: {e}")
            return False
    elif gemini_api_available and not GEMINI_API_KEY:
        print("FEHLER: Gemini API Key fehlt. API kann nicht konfiguriert werden.")
        return False
    else:
        # print("Info: Gemini API SDK nicht verfügbar.")
        return False

def translate_module_activation(activation: float) -> str:
    """Übersetzt numerische Modulaktivierung in eine beschreibende Stufe."""
    if activation >= 0.75: return "hoch"
    elif activation >= 0.45: return "mittel"
    else: return "niedrig"

# --- Schritt 1: Input-Daten generieren ---
def get_input_data(user_prompt: str, status_callback: Callable[[str], None]) -> Optional[pd.DataFrame]:
    """
    Ruft die (simulierte) Gemini Perception Unit auf, um das Eingabe-DataFrame zu erzeugen.
    """
    status_callback("Generiere Input-Daten (Simulation)...")
    try:
        input_df = generate_prompt_based_csv(user_prompt)
        if input_df is None or input_df.empty:
            status_callback("FEHLER: Input-Daten Generierung fehlgeschlagen (leeres Ergebnis).")
            print("FEHLER: generate_prompt_based_csv hat None oder ein leeres DataFrame zurückgegeben.")
            return None
        status_callback(f"{len(input_df)} Input-Einträge generiert.")
        return input_df
    except ValueError as ve:
        status_callback(f"FEHLER bei Input-Generierung: {ve}")
        print(f"FEHLER in generate_prompt_based_csv: {ve}")
        return None
    except Exception as e:
        status_callback(f"FEHLER bei Input-Generierung: {e}")
        print(f"Unerwarteter FEHLER während get_input_data: {e}")
        import traceback
        traceback.print_exc()
        return None

# --- Schritt 2: NeuroPersona Simulation ---
def run_neuropersona(input_df: pd.DataFrame, params: Dict[str, Any], status_callback: Callable[[str], None]) -> Optional[Tuple[str, Dict]]:
    """
    Führt die NeuroPersona Simulation durch.
    """
    status_callback("Starte NeuroPersona Simulation...")
    try:
        results = run_neuropersona_simulation(
            input_df=input_df,
            epochs=params.get('epochs', DEFAULT_EPOCHS),
            learning_rate=params.get('learning_rate', DEFAULT_LEARNING_RATE),
            decay_rate=params.get('decay_rate', DEFAULT_DECAY_RATE),
            reward_interval=params.get('reward_interval', DEFAULT_REWARD_INTERVAL),
            generate_plots=params.get('generate_plots', True),
            save_state=params.get('save_state', False)
        )

        if results is None or results[0] is None or results[1] is None:
            status_callback("FEHLER: NeuroPersona Simulation fehlgeschlagen (keine Ergebnisse).")
            print("FEHLER: run_neuropersona_simulation hat None oder ungültige Ergebnisse zurückgegeben.")
            return None

        report_text, structured_results = results
        status_callback("NeuroPersona Simulation abgeschlossen.")
        # print("\n--- NeuroPersona Strukturierte Ergebnisse (für Debugging) ---")
        # print(json.dumps(structured_results, indent=2, ensure_ascii=False))
        # print("-" * 50)
        return report_text, structured_results

    except Exception as e:
        status_callback(f"FEHLER während NeuroPersona: {e}")
        print(f"Unerwarteter FEHLER während run_neuropersona: {e}")
        import traceback
        traceback.print_exc()
        return None

# --- Schritt 3: Finale Antwort mit Gemini API ---
def generate_final_response(original_user_prompt: str, neuropersona_report: str, structured_results: Dict, status_callback: Callable[[str], None]) -> Optional[str]:
    """
    Generiert die finale Antwort mithilfe der echten Gemini API,
    basierend auf dem Prompt und den NeuroPersona Ergebnissen.
    (Verwendet String-Konkatenation statt eines langen Triple-Quote-Strings)
    """
    status_callback("Konfiguriere Gemini API für finale Antwort...")
    if not configure_gemini_api():
        error_msg = "FEHLER: Gemini API nicht verfügbar oder nicht konfiguriert. Finale Antwort kann nicht generiert werden."
        status_callback(error_msg)
        print(error_msg)
        return None

    status_callback("Erstelle Prompt für finale Gemini-Analyse...")

    dominant_category = structured_results.get('dominant_category', 'Unbekannt')
    dominant_activation = structured_results.get('dominant_activation', 0.0)
    module_activations = structured_results.get('module_activations', {})

    creativus_level = translate_module_activation(module_activations.get("Cortex Creativus", 0))
    criticus_level = translate_module_activation(module_activations.get("Cortex Criticus", 0))
    limbus_level = translate_module_activation(module_activations.get("Limbus Affektus", 0))
    meta_level = translate_module_activation(module_activations.get("Meta Cognitio", 0))

    # --- Prompt aus Teilen zusammensetzen ---
    prompt_parts = []
    prompt_parts.append("Du bist ein spezialisierter Analyse-Assistent. Deine Aufgabe ist es, die Ergebnisse einer komplexen, bio-inspirierten Simulation (NeuroPersona) zu interpretieren und darauf basierend eine prägnante, aufschlussreiche und stilistisch angepasste Antwort auf die ursprüngliche Benutzerfrage zu formulieren. Am Ende musst du dann eine zusätzlich ausführliche Antwort ohne Erklärung basierend auf den analysedaten erstellen und dich daran halten ohne NeuroPersona-Simulation in dieser Antwort zu erwähnen! So als wäre es deine eigene antwort!")
    prompt_parts.append("\n\n**1. Ursprüngliche Benutzerfrage:**")
    prompt_parts.append(f'"{original_user_prompt}"') # f-string hier ok, da kurz
    prompt_parts.append("\n\n**2. Analysebericht der NeuroPersona Simulation:**")
    prompt_parts.append("Dieser Bericht fasst den Endzustand des simulierten neuronalen Netzwerks zusammen. Beachte die Tendenzen der Kategorien und den Zustand der kognitiven Module.")
    prompt_parts.append("```text")
    prompt_parts.append(neuropersona_report) # Bericht als separater Teil
    prompt_parts.append("```")
    prompt_parts.append("\n**3. Wichtige extrahierte Ergebnisse & \"Persönlichkeit\" der Simulation:**")
    prompt_parts.append(f"*   **Hauptfokus (Dominante Kategorie):** {dominant_category} (Aktivierung: {dominant_activation:.3f})")
    prompt_parts.append(f"*   **Kreativitätslevel (Cortex Creativus):** {creativus_level}")
    prompt_parts.append(f"*   **Kritiklevel (Cortex Criticus):** {criticus_level}")
    prompt_parts.append(f"*   **Emotionalitätslevel (Limbus Affektus):** {limbus_level}")
    prompt_parts.append(f"*   **Strategielevel (Meta Cognitio):** {meta_level}")
    prompt_parts.append("\n**4. Deine Anweisungen für die finale Antwort (SEHR WICHTIG):**")
    # Verwende .format() für den String mit geschweiften Klammern innen drin
    prompt_parts.append("\n*   **Fokus:** Deine Antwort muss sich klar auf die **dominante Kategorie '{dominant_category}'** konzentrieren. Interpretiere, was die Aktivierung dieser Kategorie im Kontext der Benutzerfrage bedeutet. Andere Kategorien aus dem Bericht können unterstützend erwähnt werden, aber der Hauptfokus liegt auf der dominanten.".format(dominant_category=dominant_category))
    prompt_parts.append("*   **Stil-Anpassung (entscheidend!):** Passe den Tonfall und die Art deiner Antwort an die \"Persönlichkeit\" der Simulation (Modullevel) an:")
    prompt_parts.append("    *   **Hoher Kreativitätslevel:** Sei spekulativer, denke \"out-of-the-box\", schlage originelle Ideen oder unerwartete Verbindungen vor (immer im Kontext der dominanten Kategorie).")
    prompt_parts.append("    *   **Hoher Kritiklevel:** Sei vorsichtiger, betone Unsicherheiten, Risiken, alternative Sichtweisen oder potenzielle Fallstricke. Formuliere zurückhaltender.")
    prompt_parts.append("    *   **Hoher Emotionalitätslevel:** Nutze eine etwas emotionalere Sprache (z.B. \"besonders vielversprechend\", \"beunruhigend\"), aber bleibe professionell. Spiegele die zugrundeliegende positive/negative Tönung wider, falls im Bericht erkennbar.")
    prompt_parts.append("    *   **Hoher Strategielevel:** Biete einen strategischen Ausblick, leite Handlungsempfehlungen ab, denke über langfristige Implikationen nach oder strukturiere die Antwort sehr logisch.")
    prompt_parts.append("    *   **Kombinationen:** Wenn mehrere Module hoch aktiviert sind, kombiniere die Stile (z.B. hoch Creativus + hoch Criticus -> \"kreativ-kritisch\", d.h. innovative Ideen mit klarem Blick für Risiken). Bei mittleren/niedrigen Leveln -> neutraler, sachlicher Stil.")
    prompt_parts.append("*   **Basis:** Deine Argumentation **MUSS** auf den Erkenntnissen des NeuroPersona-Berichts basieren. Erfinde keine Fakten. Interpretiere die Simulation, füge aber keine externen Informationen hinzu, die dem Bericht widersprechen.")
    prompt_parts.append("*   **Integration:** Verwebe die Simulationsergebnisse (insbesondere zur dominanten Kategorie und den Modulen) **natürlich** in deine Antwort auf die ursprüngliche Benutzerfrage. Zitiere nicht nur, sondern **synthetisiere** die Informationen zu einer kohärenten Aussage.")
    prompt_parts.append("*   **Format:** Gib eine gut lesbare, prägnante Antwort in deutscher Sprache im **Markdown-Format** aus. Sprich den Nutzer ggf. direkt an (\"Ihre Frage bezüglich...\", \"Die Simulation deutet darauf hin...\"). Vermeide es, dich selbst als \"Gemini\" oder \"KI\" zu bezeichnen, präsentiere dich als Analyse-Assistent.")
    prompt_parts.append("\nGeneriere jetzt die finale, aufbereitete Antwort für den Benutzer:")

    # Alle Teile zu einem String zusammenfügen
    final_prompt = "\n".join(prompt_parts)

    status_callback(f"Sende Anfrage an Gemini API ({FINAL_RESPONSE_MODEL_NAME})...")
    try:
        model = genai.GenerativeModel(FINAL_RESPONSE_MODEL_NAME)
        safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        ]
        generation_config = genai.types.GenerationConfig(
             max_output_tokens=2048
            )

        response = model.generate_content(
            final_prompt, # Der zusammengefügte Prompt
            generation_config=generation_config,
            safety_settings=safety_settings
            )

        status_callback("Antwort von Gemini API erhalten.")

        try:
            final_answer_text = response.text
        except ValueError:
            print("WARNUNG: Gemini API hat die Antwort möglicherweise blockiert.")
            print(f"Blockierungsgrund (Feedback): {response.prompt_feedback}")
            try:
                final_answer_text = "".join(part.text for part in response.parts)
                if not final_answer_text:
                    final_answer_text = f"FEHLER: Gemini API hat die Antwort blockiert. Grund: {response.prompt_feedback}"
            except Exception:
                 final_answer_text = f"FEHLER: Gemini API hat die Antwort blockiert und Teile konnten nicht extrahiert werden. Grund: {response.prompt_feedback}"

        final_answer_text = final_answer_text.strip()
        return final_answer_text

    except Exception as e:
        status_callback(f"FEHLER bei Gemini API-Aufruf: {e}")
        print(f"FEHLER bei der Kommunikation mit der Gemini API: {e}")
        import traceback
        traceback.print_exc()
        return f"FEHLER bei der Generierung der finalen Antwort durch Gemini: {e}"

# --- Haupt-Workflow Funktion ---
def execute_full_workflow(
    user_prompt: str,
    neuropersona_epochs: int = DEFAULT_EPOCHS,
    neuropersona_lr: float = DEFAULT_LEARNING_RATE,
    neuropersona_dr: float = DEFAULT_DECAY_RATE,
    neuropersona_ri: int = DEFAULT_REWARD_INTERVAL,
    neuropersona_gen_plots: bool = True,
    neuropersona_save_state: bool = False,
    neuropersona_load_state: bool = False,  # <-- HINZUFÜGEN
    status_callback: Callable[[str], None] = _default_status_callback
) -> Optional[str]:

    """
    Führt den gesamten 3-Schritte-Workflow aus.
    """
    start_time_workflow = time.time()
    status_callback(f"Workflow gestartet für Prompt: '{user_prompt[:50]}...'")
    final_response = None

    # Schritt 1: Input-Daten generieren (Simulation)
    input_df = get_input_data(user_prompt, status_callback)
    if input_df is None:
        status_callback("Workflow abgebrochen (Fehler bei Input-Generierung).")
        return "FEHLER: Konnte keine Input-Daten für NeuroPersona generieren."

    # Schritt 2: NeuroPersona Simulation durchführen
    neuropersona_params = {
        'epochs': neuropersona_epochs,
        'learning_rate': neuropersona_lr,
        'decay_rate': neuropersona_dr,
        'reward_interval': neuropersona_ri,
        'generate_plots': neuropersona_gen_plots,
        'save_state': neuropersona_save_state,
        'load_state': neuropersona_load_state  # <-- HINZUFÜGEN
    }

    neuropersona_results = run_neuropersona(input_df, neuropersona_params, status_callback)
    if neuropersona_results is None:
        status_callback("Workflow abgebrochen (Fehler bei NeuroPersona Simulation).")
        return "FEHLER: NeuroPersona Simulation ist fehlgeschlagen."

    neuropersona_report_text, structured_results = neuropersona_results

    # Schritt 3: Finale Antwort mit echter Gemini API generieren
    if neuropersona_report_text is None: neuropersona_report_text = "NeuroPersona-Bericht konnte nicht generiert werden."
    if structured_results is None: structured_results = {}

    final_response = generate_final_response(user_prompt, neuropersona_report_text, structured_results, status_callback)

    if final_response is None:
         status_callback("Workflow beendet (Fehler bei finaler API-Kommunikation).")
         fallback_message = "FEHLER: Konnte keine finale Antwort von Gemini erhalten (API-Problem oder Konfiguration)."
         return f"{fallback_message}\n\n--- NeuroPersona Roh-Analyse: ---\n{neuropersona_report_text}"
    elif "FEHLER" in final_response:
        status_callback(f"Workflow beendet (Fehler in finaler Antwort: {final_response[:100]}...).")
        return f"{final_response}\n\n--- NeuroPersona Roh-Analyse: ---\n{neuropersona_report_text}"
    else:
        status_callback("Workflow erfolgreich abgeschlossen.")

    end_time_workflow = time.time()
    print(f"Gesamter Workflow dauerte {end_time_workflow - start_time_workflow:.2f} Sekunden.")

    return final_response

# --- Testaufruf für das Orchestrator-Skript ---
if __name__ == "__main__":
    print("--- Starte NeuroPersona Workflow Orchestrator (Direktaufruf) ---")

    if not GEMINI_API_KEY and gemini_api_available:
        print("\nACHTUNG: GEMINI_API_KEY ist nicht gesetzt. Schritt 3 (Finale Antwort) wird fehlschlagen.")
        print("Bitte setzen Sie die Umgebungsvariable, z.B.: export GEMINI_API_KEY='IHR_API_SCHLUESSEL'\n")

    initial_user_prompt = input("Geben Sie Ihre Analysefrage ein (z.B. 'Zukunft der Elektromobilität'): ")

    if initial_user_prompt:
        final_answer = execute_full_workflow(
            initial_user_prompt,
            neuropersona_epochs=25,
            neuropersona_lr=0.08,
            neuropersona_gen_plots=True,
            neuropersona_save_state=False
        )

        print("\n" + "="*50)
        print(">>> Finale Antwort des Workflows: <<<")
        print("="*50)
        if final_answer:
            print(final_answer)
        else:
            print("Workflow konnte keine finale Antwort produzieren (siehe Fehlermeldungen oben).")
        print("="*50)
    else:
        print("Keine Eingabe, Workflow nicht gestartet.")