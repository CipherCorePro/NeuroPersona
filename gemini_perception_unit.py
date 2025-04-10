# -*- coding: utf-8 -*-
# Filename: gemini_perception_unit.py
"""
GeminiPerceptionUnit

Dieses Modul wandelt einen beliebigen User-Prompt dynamisch in eine strukturierte
CSV-ähnliche Tabelle (Frage, Antwort, Kategorie) um. Es nutzt aktuell eine Simulation,
um die API-Anfrage für die CSV-Generierung zu simulieren.
"""

import pandas as pd
import random
import re # Importiere Regular Expressions für die Prompt-Analyse

# --- Konstante Minimalanzahl ---
MIN_ENTRIES = 12 # Leicht erhöht für mehr Vielfalt

# --- Simulation für Testzwecke ---
def simulate_gemini_response(user_prompt: str, min_entries: int = MIN_ENTRIES) -> list:
    """
    Simuliert eine Gemini-Antwort mit dynamischer Erzeugung von (Frage, Antwort, Kategorie),
    die thematisch zum User-Prompt passt (vereinfachte Heuristik).
    """
    # Extrahiere Schlüsselwörter oder das Hauptthema (sehr einfache Methode)
    keywords = re.findall(r'\b\w{4,}\b', user_prompt) # Finde Wörter mit mind. 4 Buchstaben
    theme = keywords[0] if keywords else "das Thema" # Nimm das erste längere Wort als Thema

    # Generiere vielfältigere Fragen basierend auf dem Thema
    question_templates = [
        f"Was sind die Hauptmerkmale von {theme}?",
        f"Welche Chancen bietet {theme}?",
        f"Welche Risiken sind mit {theme} verbunden?",
        f"Wie beeinflusst {theme} den Bereich X?", # Bereich X könnte spezifischer sein
        f"Gibt es ethische Bedenken bei {theme}?",
        f"Was sind typische Anwendungsfälle für {theme}?",
        f"Wie unterscheidet sich {theme} von verwandten Konzepten?",
        f"Welche Kosten sind mit {theme} verbunden?",
        f"Wie ist die aktuelle Marktsituation für {theme}?",
        f"Welche Prognosen gibt es für die Entwicklung von {theme}?",
        f"Gibt es regulatorische Aspekte bei {theme}?",
        f"Welche Fähigkeiten benötigt man im Umgang mit {theme}?",
        f"Wie ist die öffentliche Wahrnehmung von {theme}?",
        f"Welche Innovationen treiben {theme} voran?",
    ]

    # Generiere passendere Antworten (immer noch generisch, aber leicht variiert)
    answer_templates = [
        "Die Merkmale umfassen A, B und C.",
        "Wesentliche Chancen liegen in der Effizienzsteigerung und neuen Märkten.",
        "Risiken beinhalten hohe Anfangskosten und Akzeptanzprobleme.",
        f"{theme.capitalize()} revolutioniert diesen Bereich durch Automatisierung.",
        "Ja, Datenschutz und Bias sind zentrale ethische Fragen.",
        "Anwendungen finden sich oft in Industrie und Forschung.",
        f"Im Gegensatz zu Y fokussiert {theme} stärker auf Z.",
        "Die Kosten variieren stark, oft im sechsstelligen Bereich.",
        "Der Markt wächst schnell, ist aber fragmentiert.",
        "Prognosen deuten auf exponentielles Wachstum in den nächsten 5 Jahren hin.",
        "Regulierungen sind noch in Entwicklung, aber wichtig.",
        "Analytische Fähigkeiten und Datenkompetenz sind entscheidend.",
        "Die öffentliche Meinung ist gespalten, aber tendenziell neugierig.",
        "Innovationen kommen vor allem aus der KI-Forschung und Hardware-Verbesserungen."
    ]

    # Kategorienset
    categories = [
        "Grundlagen", "Chancen", "Risiken", "Auswirkungen", "Ethik", "Anwendung",
        "Vergleich", "Kosten", "Markt", "Prognose", "Regulierung", "Fähigkeiten",
        "Wahrnehmung", "Innovation", "Technologie", "Wirtschaft"
    ]

    generated = []
    # Stelle sicher, dass mindestens min_entries generiert werden
    num_to_generate = max(min_entries, len(question_templates))

    for i in range(num_to_generate):
        # Wähle zufällig, aber versuche Wiederholungen zu vermeiden (einfacher Ansatz)
        q = random.choice(question_templates)
        a = random.choice(answer_templates)
        c = random.choice(categories)

        # Leichte Anpassung der Antworten basierend auf der Frage (sehr simpel)
        if "risiken" in q.lower():
            a = random.choice([ans for ans in answer_templates if "risik" in ans.lower() or "kosten" in ans.lower() or "problem" in ans.lower()] or [a])
            c = "Risiken"
        elif "chancen" in q.lower():
            a = random.choice([ans for ans in answer_templates if "chance" in ans.lower() or "vorteil" in ans.lower() or "wachstum" in ans.lower()] or [a])
            c = "Chancen"
        elif "merkmal" in q.lower() or "definition" in q.lower():
             c = "Grundlagen"
        elif "prognose" in q.lower() or "zukunft" in q.lower():
             c = "Prognose"
        # ... weitere einfache Regeln hinzufügen ...

        generated.append({"Frage": q, "Antwort": a, "Kategorie": c})

    # Entferne exakte Duplikate (basierend auf Frage UND Antwort)
    unique_generated = []
    seen = set()
    for item in generated:
        identifier = (item["Frage"], item["Antwort"])
        if identifier not in seen:
            unique_generated.append(item)
            seen.add(identifier)

    # Falls nach Deduplizierung zu wenige Einträge, fülle auf (selten nötig)
    while len(unique_generated) < min_entries:
         q = random.choice(question_templates)
         a = random.choice(answer_templates)
         c = random.choice(categories)
         identifier = (q, a)
         if identifier not in seen:
              unique_generated.append({"Frage": q, "Antwort": a, "Kategorie": c})
              seen.add(identifier)


    return unique_generated[:num_to_generate] # Stelle sicher, nicht mehr als gewünscht zurückzugeben

# --- Platzhalter für echte API-Abfrage (nicht benötigt für Simulation) ---
# def query_gemini_api(user_prompt: str) -> str:
#     """(Optional) Echte API-Anfrage."""
#     # Hier wäre der API-Call
#     return "Dummy Gemini Response"

# --- Ausgabe parsen ---
def parse_gemini_output_to_dataframe(gemini_output: list) -> pd.DataFrame:
    """
    Wandelt eine Gemini-Antwortliste (Liste von Dictionaries) in ein DataFrame.
    """
    if not isinstance(gemini_output, list) or not gemini_output:
        raise ValueError("Leere oder ungültige Gemini-Ausgabe erhalten (Liste erwartet).")

    try:
        df = pd.DataFrame(gemini_output)
    except Exception as e:
        raise ValueError(f"Fehler beim Erstellen des DataFrame aus der Liste: {e}")


    # Sicherheitschecks auf Spaltennamen
    expected_cols = {"Frage", "Antwort", "Kategorie"}
    if not expected_cols.issubset(df.columns):
        missing_cols = expected_cols - set(df.columns)
        raise ValueError(f"Erwartete Spalten fehlen im DataFrame: {missing_cols}")

    # Optional: Prüfe auf leere Werte (NaN) und fülle sie ggf. auf
    if df.isnull().values.any():
        print("Warnung: DataFrame enthält fehlende Werte (NaN).")
        # df.fillna("Unbekannt", inplace=True) # Beispiel: Mit "Unbekannt" füllen

    return df

# --- Hauptfunktion für den gesamten Ablauf ---
def generate_prompt_based_csv(user_prompt: str, min_entries: int = MIN_ENTRIES) -> pd.DataFrame:
    """
    Hauptfunktion: Erzeugt DataFrame aus User-Prompt mittels Simulation.
    """
    if not user_prompt or not user_prompt.strip():
        raise ValueError("Benutzereingabe (User-Prompt) ist leer.")

    print(f"Erzeuge SIMULIERTE Input-Daten für: '{user_prompt}'...")

    # Rufe die Simulationsfunktion auf
    simulated_response_list = simulate_gemini_response(user_prompt, min_entries)

    # Wandle die Liste von Dictionaries in ein DataFrame um
    df = parse_gemini_output_to_dataframe(simulated_response_list)

    print(f"Simulierte Input-Daten generiert ({len(df)} Einträge).")
    return df

# --- Testaufruf (nur für Direktstart des Skripts) ---
if __name__ == "__main__":
    print("--- Testlauf: Gemini Perception Unit (Simulation) ---")
    # Beispiel-Prompts
    prompts = ["künstliche Intelligenz in der Medizin", "Auswirkungen des Klimawandels auf die Landwirtschaft", "Tesla Aktie Prognose 2025"]
    selected_prompt = random.choice(prompts)
    print(f"Teste mit Prompt: '{selected_prompt}'")

    try:
        df_result = generate_prompt_based_csv(selected_prompt)
        print("\n--- Ergebnis DataFrame (erste 10 Zeilen): ---")
        # to_string für bessere Konsolenausgabe, zeigt mehr Spaltenbreite
        print(df_result.head(10).to_string())
        print("\n--- DataFrame Info: ---")
        df_result.info()
    except ValueError as e:
        print(f"\nFEHLER im Testlauf: {e}")
    except Exception as e:
        print(f"\nUnerwarteter FEHLER im Testlauf: {e}")