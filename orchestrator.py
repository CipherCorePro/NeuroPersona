import os
import time
import pandas as pd
from typing import Callable, Optional, Tuple, Dict, Any
import json
import traceback

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
FINAL_RESPONSE_MODEL_NAME = 'gemini-1.5-flash-latest'

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
    from neuropersona_core import run_neuropersona_simulation, DEFAULT_EPOCHS as NP_DEFAULT_EPOCHS, DEFAULT_LEARNING_RATE as NP_DEFAULT_LR, DEFAULT_DECAY_RATE as NP_DEFAULT_DR, DEFAULT_REWARD_INTERVAL as NP_DEFAULT_RI
    DEFAULT_EPOCHS = NP_DEFAULT_EPOCHS
    DEFAULT_LEARNING_RATE = NP_DEFAULT_LR
    DEFAULT_DECAY_RATE = NP_DEFAULT_DR
    DEFAULT_REWARD_INTERVAL = NP_DEFAULT_RI
except ImportError as e:
    print(f"FEHLER: Notwendige Skripte (gemini_perception_unit.py, neuropersona_core.py) nicht gefunden oder Importfehler: {e}")
    print("Stellen Sie sicher, dass die Dateien im aktuellen Verzeichnis oder im PYTHONPATH vorhanden sind.")
    exit()

def _default_status_callback(message: str):
    print(f"[Orchestrator Status] {message}")

def configure_gemini_api():
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
        return False

def translate_module_activation(activation: float) -> str:
    if activation >= 0.75: return "hoch"
    elif activation >= 0.45: return "mittel"
    else: return "niedrig"

def get_input_data(user_prompt: str, status_callback: Callable[[str], None]) -> Optional[pd.DataFrame]:
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
        traceback.print_exc()
        return None

def run_neuropersona(input_df: pd.DataFrame, params: Dict[str, Any], status_callback: Callable[[str], None]) -> Optional[Tuple[str, Dict]]:
    status_callback("Starte NeuroPersona Simulation...")
    try:
        results = run_neuropersona_simulation(
            input_df=input_df,
            epochs=params.get('epochs', DEFAULT_EPOCHS),
            learning_rate=params.get('learning_rate', DEFAULT_LEARNING_RATE),
            decay_rate=params.get('decay_rate', DEFAULT_DECAY_RATE),
            reward_interval=params.get('reward_interval', DEFAULT_REWARD_INTERVAL),
            generate_plots=params.get('generate_plots', True),
            save_state=params.get('save_state', False),
            load_state=params.get('load_state', False)
        )

        if results is None or not isinstance(results, tuple) or len(results) != 2:
            status_callback("FEHLER: NeuroPersona Simulation fehlgeschlagen (ungültige Ergebnisse).")
            print("FEHLER: run_neuropersona_simulation hat keine gültige (report_text, structured_results) Tuple zurückgegeben.")
            return None

        report_text, structured_results = results

        if report_text is None or structured_results is None:
             status_callback("FEHLER: NeuroPersona Simulation fehlgeschlagen (Teilergebnisse sind None).")
             print("FEHLER: run_neuropersona_simulation hat None als report_text oder structured_results zurückgegeben.")
             return report_text if report_text is not None else "", structured_results if structured_results is not None else {}

        status_callback("NeuroPersona Simulation abgeschlossen.")
        return report_text, structured_results

    except Exception as e:
        status_callback(f"FEHLER während NeuroPersona: {e}")
        print(f"Unerwarteter FEHLER während run_neuropersona: {e}")
        traceback.print_exc()
        return None

def generate_final_response(original_user_prompt: str, neuropersona_report: str, structured_results: Dict, status_callback: Callable[[str], None]) -> Optional[str]:
    status_callback("Konfiguriere Gemini API für finale Antwort...")
    if not configure_gemini_api():
        error_msg = "FEHLER: Gemini API nicht verfügbar oder nicht konfiguriert. Finale Antwort kann nicht generiert werden."
        status_callback(error_msg)
        print(error_msg)
        return None

    status_callback("Erstelle Prompt für finale Gemini-Antwort...")

    if structured_results is None:
        structured_results = {}

    dominant_category = structured_results.get('dominant_category', 'Unbekannt')
    dominant_activation = structured_results.get('dominant_activation', 0.0)
    module_activations = structured_results.get('module_activations', {})

    if not isinstance(module_activations, dict):
        module_activations = {}

    creativus_level = translate_module_activation(module_activations.get("Cortex Creativus", 0.0))
    criticus_level = translate_module_activation(module_activations.get("Cortex Criticus", 0.0))
    limbus_level = translate_module_activation(module_activations.get("Limbus Affektus", 0.0))
    meta_level = translate_module_activation(module_activations.get("Meta Cognitio", 0.0))

    prompt_parts = []
    prompt_parts.append("Du bist ein hilfreicher Assistent. Deine Aufgabe ist es, direkt auf die ursprüngliche Benutzerfrage zu antworten. Nutze die bereitgestellten Analyse-Ergebnisse (NeuroPersona) als *internen Leitfaden*, um den Stil, Ton und gegebenenfalls den Inhalt deiner Antwort subtil anzupassen. **Erkläre die Analyse selbst, die Simulationsdetails oder die Modulnamen NICHT in deiner finalen Antwort an den Benutzer.** Formuliere einfach eine passende, natürliche Antwort auf die Frage des Nutzers.")

    prompt_parts.append("\n\n**1. Ursprüngliche Benutzerfrage:**")
    prompt_parts.append(f'"{original_user_prompt}"')

    prompt_parts.append("\n\n**2. Kontext: Analysebericht einer Simulation (Nur für deine interne Orientierung):**")
    prompt_parts.append("```text")
    prompt_parts.append(str(neuropersona_report) if neuropersona_report is not None else "Kein Bericht verfügbar.")
    prompt_parts.append("```")

    prompt_parts.append("\n**3. Kontext: Wichtige extrahierte Ergebnisse & \"Persönlichkeit\" der Simulation (Nur für deine interne Orientierung):**")
    prompt_parts.append(f"*   **Hauptfokus (Dominante Kategorie):** {dominant_category} (Aktivierung: {dominant_activation:.3f})")
    prompt_parts.append(f"*   **Kreativitätslevel (Cortex Creativus):** {creativus_level}")
    prompt_parts.append(f"*   **Kritiklevel (Cortex Criticus):** {criticus_level}")
    prompt_parts.append(f"*   **Emotionalitätslevel (Limbus Affektus):** {limbus_level}")
    prompt_parts.append(f"*   **Strategielevel (Meta Cognitio):** {meta_level}")

    prompt_parts.append("\n**4. Deine Anweisungen für die finale Antwort (SEHR WICHTIG):**")
    prompt_parts.append("*   **Kernaufgabe:** Formuliere eine **direkte, natürliche Antwort** auf die ursprüngliche Benutzerfrage.")
    prompt_parts.append("*   **Analyse-Einfluss (Subtil anwenden!):**")
    prompt_parts.append("    *   Lass die **dominante Kategorie ('{dom_cat}')** den thematischen Unterton deiner Antwort beeinflussen, aber ohne sie explizit zu nennen. Beispiel: Bei 'Kosten' könnte eine einfache Frage wie 'Hallo' knapper oder zielgerichteter beantwortet werden (z.B. 'Hallo. Was gibt's?'), bei einer komplexen Frage könnten kostenbezogene Aspekte subtil anklingen.".format(dom_cat=dominant_category))
    prompt_parts.append("    *   Passe den **Tonfall und Stil** deiner *direkten Antwort* an die Modullevel an:")
    prompt_parts.append("        *   **Hoher Kreativitätslevel:** Antwort kann etwas origineller, unerwarteter, bildhafter oder assoziativer sein.")
    prompt_parts.append("        *   **Hoher Kritiklevel:** Antwort kann vorsichtiger, abwägender, nachfragender sein oder potenzielle Probleme/Alternativen andeuten (z.B. 'Interessanter Punkt, allerdings sollte man auch bedenken...').")
    prompt_parts.append("        *   **Hoher Emotionalitätslevel:** Antwort kann etwas wärmer, empathischer oder (bei negativer Tönung) zurückhaltender/besorgter klingen, aber bleibe professionell.")
    prompt_parts.append("        *   **Hoher Strategielevel:** Antwort kann strukturierter, zielorientierter, lösungsorientierter oder vorausschauender sein.")
    prompt_parts.append("        *   **Kombinationen/Neutral:** Kombiniere Stile bei mehreren hohen Aktivierungen oder sei neutral/sachlich/standardmäßig bei niedrigen/mittleren Leveln.")
    prompt_parts.append("*   **KEINE Analyse-Erklärung:** Erwähne **NIEMALS** die Simulation, NeuroPersona, Kategorien (wie '{dom_cat}'), Modullevel (wie 'hoch Kreativus'), interne Prozesse oder den Analysebericht in deiner Antwort an den Benutzer. Deine Antwort soll wie eine ganz normale, passende Reaktion wirken.".format(dom_cat=dominant_category))
    prompt_parts.append("*   **Inspiration:** Lass dich von den Analyseergebnissen inspirieren, aber erfinde keine Fakten und bleibe im Rahmen dessen, was als Antwort auf die ursprüngliche Nutzerfrage sinnvoll ist.")
    prompt_parts.append("*   **Format:** Gib NUR die **direkte Antwort** auf die Benutzerfrage in deutscher Sprache im **Markdown-Format** aus. Sprich den Nutzer natürlich an. Vermeide Selbstbezeichnungen wie 'KI' oder 'Assistent'. Konzentriere dich auf die Antwort selbst. Etscheide selbständig anhand der analyse ergebnisse wie lang deine Antwort seien sollte!")

    prompt_parts.append("\nGeneriere jetzt die finale, direkte und analyse-informierte Antwort für den Benutzer und endscheide wie lang diese anhand der analysedaten sein muss!:")

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
             temperature=0.7
            )

        response = model.generate_content(
            final_prompt,
            generation_config=generation_config,
            safety_settings=safety_settings
            )

        status_callback("Antwort von Gemini API erhalten.")

        final_answer_text = ""
        try:
            final_answer_text = response.text
        except ValueError:
            print("WARNUNG: Gemini API hat die Antwort möglicherweise blockiert.")
            print(f"Blockierungsgrund (Feedback): {response.prompt_feedback}")
            try:
                final_answer_text = "".join(part.text for part in response.parts)
                if not final_answer_text:
                     final_answer_text = f"FEHLER: Gemini API hat die Antwort blockiert. Grund: {response.prompt_feedback}"
            except Exception as e_parts:
                 print(f"Konnte auch keine Teile extrahieren nach Blockierung: {e_parts}")
                 final_answer_text = f"FEHLER: Gemini API hat die Antwort blockiert und Teile konnten nicht extrahiert werden. Grund: {response.prompt_feedback}"
        except Exception as e_text:
            print(f"Unerwarteter Fehler beim Zugriff auf response.text: {e_text}")
            final_answer_text = f"FEHLER: Unerwarteter Fehler beim Verarbeiten der Gemini-Antwort: {e_text}"

        final_answer_text = final_answer_text.strip()

        return final_answer_text

    except genai.types.generation_types.BlockedPromptException as bpe:
         status_callback(f"FEHLER: Der Prompt wurde von der Gemini API blockiert: {bpe}")
         print(f"FEHLER: Der Prompt wurde von der Gemini API blockiert: {bpe}")
         print(f"Blockierungsgrund (Feedback): {response.prompt_feedback if 'response' in locals() else 'Kein Response-Objekt verfügbar'}")
         return f"FEHLER: Der Prompt wurde von der Gemini API blockiert. Grund: {response.prompt_feedback if 'response' in locals() else 'Unbekannt'}"
    except Exception as e:
        status_callback(f"FEHLER bei Gemini API-Aufruf: {e}")
        print(f"FEHLER bei der Kommunikation mit der Gemini API: {e}")
        traceback.print_exc()
        return f"FEHLER bei der Generierung der finalen Antwort durch Gemini: {e}"

def execute_full_workflow(
    user_prompt: str,
    neuropersona_epochs: int = DEFAULT_EPOCHS,
    neuropersona_lr: float = DEFAULT_LEARNING_RATE,
    neuropersona_dr: float = DEFAULT_DECAY_RATE,
    neuropersona_ri: int = DEFAULT_REWARD_INTERVAL,
    neuropersona_gen_plots: bool = True,
    neuropersona_save_state: bool = False,
    neuropersona_load_state: bool = False,
    status_callback: Callable[[str], None] = _default_status_callback
) -> Optional[str]:
    start_time_workflow = time.time()
    status_callback(f"Workflow gestartet für Prompt: '{user_prompt[:50]}...'")
    final_response = None

    input_df = get_input_data(user_prompt, status_callback)
    if input_df is None:
        status_callback("Workflow abgebrochen (Fehler bei Input-Generierung).")
        return "FEHLER: Konnte keine Input-Daten für NeuroPersona generieren."

    neuropersona_params = {
        'epochs': neuropersona_epochs,
        'learning_rate': neuropersona_lr,
        'decay_rate': neuropersona_dr,
        'reward_interval': neuropersona_ri,
        'generate_plots': neuropersona_gen_plots,
        'save_state': neuropersona_save_state,
        'load_state': neuropersona_load_state
    }

    neuropersona_results_tuple = run_neuropersona(input_df, neuropersona_params, status_callback)

    if neuropersona_results_tuple is None:
        status_callback("Workflow abgebrochen (Fehler bei NeuroPersona Simulation).")
        return "FEHLER: NeuroPersona Simulation ist fehlgeschlagen oder hat keine Ergebnisse zurückgegeben."

    neuropersona_report_text, structured_results = neuropersona_results_tuple

    neuropersona_report_text = neuropersona_report_text if neuropersona_report_text is not None else ""
    structured_results = structured_results if structured_results is not None else {}

    final_response = generate_final_response(user_prompt, neuropersona_report_text, structured_results, status_callback)

    end_time_workflow = time.time()
    status_callback(f"Workflow beendet. Gesamtdauer: {end_time_workflow - start_time_workflow:.2f} Sekunden.")

    if final_response is None:
         fallback_message = "FEHLER: Konnte keine finale Antwort generieren (API-Problem oder Konfiguration)."
         return fallback_message
    elif isinstance(final_response, str) and "FEHLER:" in final_response:
        status_callback(f"Workflow endete mit Fehler in der finalen Antwortgenerierung.")
        return final_response
    else:
        status_callback("Finale Antwort erfolgreich generiert.")
        return final_response

if __name__ == "__main__":
    print("--- Starte NeuroPersona Workflow Orchestrator (Direktaufruf) ---")

    if not gemini_api_available:
         print("\nACHTUNG: Google Generative AI SDK ist nicht installiert. Schritt 3 wird nicht funktionieren.")
    elif not GEMINI_API_KEY:
        print("\nACHTUNG: GEMINI_API_KEY ist nicht gesetzt. Schritt 3 (Finale Antwort) wird fehlschlagen.")
        print("Bitte setzen Sie die Umgebungsvariable, z.B.:")
        print("Linux/macOS: export GEMINI_API_KEY='IHR_API_SCHLUESSEL'")
        print("Windows (cmd): set GEMINI_API_KEY=IHR_API_SCHLUESSEL")
        print("Windows (PowerShell): $env:GEMINI_API_KEY='IHR_API_SCHLUESSEL'\n")

    while True:
        try:
            initial_user_prompt = input("\nGeben Sie Ihre Analysefrage ein (oder 'exit' zum Beenden): ")
            if initial_user_prompt.lower() == 'exit':
                break
            if not initial_user_prompt:
                print("Bitte geben Sie eine Frage ein.")
                continue

            final_answer = execute_full_workflow(
                initial_user_prompt,
                neuropersona_epochs=25,
                neuropersona_lr=0.08,
                neuropersona_gen_plots=False,
                neuropersona_save_state=False,
                neuropersona_load_state=False
            )

            print("\n" + "="*50)
            print(">>> Finale Antwort des Workflows: <<<")
            print("="*50)
            if final_answer:
                print(final_answer)
            else:
                print("Workflow konnte keine finale Antwort produzieren (siehe Statusmeldungen oben).")
            print("="*50)

        except EOFError:
             print("\nBeende Programm.")
             break
        except KeyboardInterrupt:
            print("\nBeende Programm.")
            break
        except Exception as e:
            print(f"\nEin unerwarteter Fehler ist im Hauptloop aufgetreten: {e}")
            traceback.print_exc()
