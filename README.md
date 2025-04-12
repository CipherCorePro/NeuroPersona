
# NeuroPersona: Simulation Dynamischer Kognitiver Perspektiven

**Version:** 1.3

## Einführung

NeuroPersona ist kein statisches Analysewerkzeug, sondern eine **bio-inspirierte Simulationsplattform**, die darauf abzielt, die dynamischen und oft variablen Prozesse menschlicher Kognition und Emotion bei der Auseinandersetzung mit einem Thema oder einer Fragestellung nachzubilden. Anstatt eine einzige, deterministische Antwort zu liefern, erforscht NeuroPersona **verschiedene plausible "Denkpfade" oder "Perspektiven"**, die als Reaktion auf einen gegebenen Input entstehen können.

Das System modelliert interagierende kognitive Module (Kreativität, Kritik, Simulation, etc.), ein adaptives Wertesystem, einen dynamischen emotionalen Zustand (PAD-Modell) und neuronale Plastizität. Jeder Simulationslauf stellt eine einzigartige Momentaufnahme eines möglichen kognitiven/affektiven Zustands dar.

## Kernphilosophie: Simulation statt Vorhersage

Der grundlegende Ansatz von NeuroPersona unterscheidet sich von traditionellen KI-Modellen:

1.  **Variabilität als Feature:** Das System ist inhärent **nicht-deterministisch**. Wiederholte Läufe mit demselben Input werden aufgrund von Faktoren wie zufälliger Initialisierung von Gewichten, stochastischem Rauschen in der Aktivierung und Emotionsdynamik sowie pfadabhängigen Lern- und Plastizitätsprozessen zu unterschiedlichen, aber intern plausiblen Endzuständen führen. Dies spiegelt die natürliche Variabilität menschlichen Denkens wider.
2.  **Emergente Perspektiven:** Jeder Simulationslauf kann als ein einzigartiger "Gedankengang" betrachtet werden, der unterschiedliche Aspekte des Themas priorisiert (mal Innovation, mal Sicherheit, mal Grundlagen). Das Ergebnis ist nicht "richtig" oder "falsch", sondern eine **simulierte, plausible Perspektive**.
3.  **Interpretation des Zustands:** Das Ziel der Analyse ist es, den **finalen kognitiven und affektiven Zustand** des Systems *innerhalb eines Laufs* zu verstehen:
    *   Welche Werte dominieren?
    *   Welche kognitiven Module sind besonders aktiv?
    *   Wie ist die emotionale Grundstimmung?
    *   Ist dieser Zustand in sich kohärent und eine nachvollziehbare Reaktion auf den Input (z.B. hohe Kreativität führt zu hohem Innovationswert)?
    *   Auch scheinbare "Inkonsistenzen" (z.B. hohe Kritikaktivität bei niedrigem Sicherheitswert) sind valide Ergebnisse, die eine bestimmte kognitive "Haltung" (z.B. Entkopplung von Analyse und Priorisierung) repräsentieren.
4.  **Erforschung des Möglichkeitsraums:** Durch die Simulation verschiedener Läufe (ggf. mit leicht variierten Parametern) kann der *Raum möglicher kognitiver Reaktionen* auf ein Thema ausgelotet werden, anstatt nur eine einzige Antwort zu generieren.

## Hauptmerkmale

*   **Dynamische Input-Verarbeitung:** Nutzt eine (simulierte) "Perception Unit", um aus Nutzer-Prompts strukturierte Daten zu generieren.
*   **Modulare Kognitive Architektur:** Simuliert interagierende Module wie:
    *   `CortexCreativus`: Ideenfindung, Assoziation.
    *   `CortexCriticus`: Analyse, Bewertung, Risikoabschätzung.
    *   `SimulatrixNeuralis`: Szenariodenken, mentale Simulation.
    *   `LimbusAffektus`: Dynamische Emotionsverarbeitung (Pleasure, Arousal, Dominance).
    *   `MetaCognitio`: Überwachung des Netzwerkzustands, adaptive Strategieanpassung (z.B. Lernrate).
    *   `CortexSocialis`: Modellierung sozialer Einflüsse.
*   **Adaptives Wertesystem:** Interne Werte (Innovation, Sicherheit, Ethik, etc.) beeinflussen das Verhalten und werden durch die Simulation dynamisch angepasst.
*   **Neuronale Plastizität:** Simulation von strukturellen Änderungen im Netzwerk (Connection Pruning & Sprouting) und aktivitätsabhängigem Lernen (Hebbian Learning, Reinforcement).
*   **Stochastizität:** Gezielter Einsatz von Zufallselementen zur Nachbildung biologischer Variabilität.
*   **Persistentes Gedächtnis:** Langfristige Speicherung und Abruf relevanter Informationen über eine SQLite-Datenbank.
*   **Reporting & Visualisierung:** Generiert detaillierte HTML-Berichte und Plots zur Analyse der Netzwerkdynamik und des Endzustands.
*   **Orchestrierung:** Ein `orchestrator.py` Skript steuert den gesamten Workflow von Prompt bis zur finalen, angereicherten Antwort (optional unter Nutzung einer externen LLM-API wie Gemini).

## Workflow Übersicht (`orchestrator.py`)

1.  **Perzeption:** Ein Nutzer-Prompt wird in strukturierte Daten (simuliertes CSV/DataFrame) umgewandelt (`gemini_perception_unit.py`).
2.  **Kognition/Simulation:** Diese Daten werden in `neuropersona_core.py` eingespeist. Das Netzwerk wird initialisiert und über eine definierte Anzahl von Epochen simuliert, wobei Lernen, Emotionen, Werte und Plastizität interagieren.
3.  **Synthese (Optional):** Die Ergebnisse der NeuroPersona-Simulation (Bericht, strukturierte Daten) werden verwendet, um eine finale, kontextualisierte Antwort zu generieren, potenziell unter Einbeziehung einer externen LLM-API (`generate_final_response` in `orchestrator.py`).

## Technische Komponenten (`neuropersona_core.py`)

*   **Klassen:** `Node`, `MemoryNode`, `ValueNode`, `Connection`, spezialisierte Modulklassen (s.o.), `PersistentMemoryManager`.
*   **Kernfunktionen:** `simulate_learning_cycle`, `calculate_value_adjustment`, `update_emotion_state`, `hebbian_learning`, `apply_reinforcement`, `prune_connections`, `sprout_connections`, `generate_final_report`, `create_html_report`, Plotting-Funktionen.
*   **Parameter:** Zahlreiche Konstanten steuern Lernraten, Zerfallsraten, Schwellwerte, Emotionsdynamik etc. und erlauben das Tuning des Systemverhaltens.

## Installation

1.  **Repository klonen:**
    ```bash
    git clone <repository-url>
    cd <repository-ordner>
    ```
2.  **Virtuelle Umgebung erstellen (empfohlen):**
    ```bash
    python -m venv venv
    # Windows
    venv\Scripts\activate
    # MacOS/Linux
    source venv/bin/activate
    ```
3.  **Abhängigkeiten installieren:**
    *(Erstelle zuerst eine `requirements.txt`, falls nicht vorhanden)*
    ```bash
    pip install -r requirements.txt
    # Benötigt mindestens: pandas, numpy, matplotlib
    # Optional: networkx, tqdm, google-generativeai
    ```
4.  **API Key (Optional):** Wenn der volle Workflow mit `orchestrator.py` und finaler Gemini-Antwort genutzt werden soll, setze den API-Key als Umgebungsvariable:
    ```bash
    # Windows (PowerShell)
    $env:GEMINI_API_KEY="DEIN_API_KEY"
    # Windows (CMD)
    set GEMINI_API_KEY=DEIN_API_KEY
    # MacOS/Linux
    export GEMINI_API_KEY='DEIN_API_KEY'
    ```

## Benutzung

Die primäre Art, eine vollständige Analyse durchzuführen, ist über die GUI oder den Orchestrator:

*   **GUI starten:**
    ```bash
    python neuropersona_core.py
    ```
    Die GUI ermöglicht die Eingabe des Prompts, die Anpassung der wichtigsten Simulationsparameter und startet den im `orchestrator.py` definierten Workflow.

*   **Orchestrator direkt aufrufen (Beispiel):**
    ```bash
    python orchestrator.py
    # Das Skript fragt nach einem Prompt, wenn es direkt ausgeführt wird.
    ```

## Interpretation der Ergebnisse

Denke an die **Kernphilosophie**:

*   **Fokus auf den Einzellauf:** Interpretiere den generierten HTML-Report und die Plots für *diesen spezifischen* Lauf.
*   **Analysiere den Zustand:** Wie verhalten sich die dominanten Kategorien, die Modulaktivitäten, die Werte und die Emotionen *zueinander*? Ist das resultierende "Profil" intern kohärent?
*   **Vergleiche nicht starr:** Erwarte keine identischen Ergebnisse bei erneuten Läufen. Beobachte stattdessen die *Bandbreite* möglicher Zustände.
*   **Sättigung (Werte bei 1.0):** Ist oft ein Zeichen für schnelles Lernen bei begrenzten Daten. Interpretiere es als "maximale Relevanz in diesem Lauf", erkenne aber an, dass die relative Differenzierung am oberen Ende verloren geht.
*   **"Inkonsistenzen" als Ergebnis:** Wenn z.B. `Cortex Criticus` hoch, aber der `Sicherheit`-Wert niedrig ist, ist das ein interpretierbares Ergebnis über die Priorisierung des Systems in diesem Lauf, kein Fehler.

## Wichtige Parameter (Konstanten in `neuropersona_core.py`)

*   `DEFAULT_EPOCHS`: Anzahl der Simulationszyklen.
*   `DEFAULT_LEARNING_RATE`: Basis-Lernrate für Gewichtsänderungen.
*   `DEFAULT_DECAY_RATE`: Wie schnell Aktivierungen/Gewichte ohne Input zerfallen (wichtig gegen Sättigung).
*   `VALUE_UPDATE_RATE`: Wie schnell sich die internen Werte anpassen.
*   `EMOTION_UPDATE_RATE`, `EMOTION_DECAY_TO_NEUTRAL`: Steuern die Dynamik der Emotionen.
*   `PRUNING_THRESHOLD`, `SPROUTING_THRESHOLD`: Steuern die strukturelle Plastizität.

Das Anpassen dieser Parameter (am besten über die GUI oder eine Einstellungsdatei) beeinflusst die **Dynamik** und **Differenzierungsfähigkeit** des Systems, nicht unbedingt die Korrektur von "Fehlern".



---
## Eine Analogie für Nicht-Wissenschaftler: Wie NeuroPersona "denkt"

Stellen Sie sich vor, Sie fragen einen Taschenrechner "Was ist 2+2?". Sie erhalten immer "4". Das ist ein **deterministisches System**.

NeuroPersona ist anders. Stellen Sie es sich eher so vor, als würden Sie einen **Menschen** eine komplexe Frage stellen, wie z.B.: "Sollten wir stark in eine neue, riskante Technologie investieren?".

*   **An Tag 1**, wenn die Person optimistisch gestimmt ist und kürzlich von Erfolgsgeschichten gehört hat, könnte die Antwort lauten: "Ja, unbedingt! Die Chancen sind riesig, wir müssen innovativ sein!" (Fokus: Innovation, Chancen).
*   **An Tag 2**, nachdem die Person von Problemen mit ähnlichen Technologien gelesen hat und vielleicht etwas besorgter ist, könnte die Antwort lauten: "Langsam! Wir müssen zuerst die Risiken genau prüfen und sicherstellen, dass alles sicher und ethisch vertretbar ist." (Fokus: Sicherheit, Ethik, Risiken).
*   **An Tag 3**, wenn die Person sehr analytisch ist, könnte sie sagen: "Lasst uns die Grundlagen analysieren und die langfristigen Auswirkungen auf die Effizienz betrachten." (Fokus: Grundlagen, Effizienz).

Alle diese Antworten sind **plausible menschliche Reaktionen**, abhängig von der inneren "Stimmung" (Emotionen), den persönlichen "Prioritäten" (Werte) und den Informationen, die gerade im Vordergrund stehen.

**NeuroPersona simuliert genau diese Art von Variabilität:**

*   Es hat interne "Stimmungen" (**Emotionen**), die sich ändern.
*   Es hat "Prioritäten" (**Werte**), die sich entwickeln.
*   Zufällige Faktoren (wie **Rauschen**) spielen eine Rolle, ähnlich wie zufällige Gedanken oder Assoziationen.
*   Der **Lernprozess** selbst beeinflusst, welche Verbindungen stärker werden und welche "Gedankenpfade" sich bilden.

**Wenn NeuroPersona also bei wiederholten Läufen unterschiedliche Ergebnisse liefert, ist das kein Fehler.** Es simuliert **unterschiedliche, aber in sich stimmige Denkweisen oder Perspektiven**, die ein kognitives System (wie ein Gehirn) als Reaktion auf die Frage entwickeln könnte. Mal ist das simulierte "Gehirn" eher optimistisch-innovativ, mal eher kritisch-vorsichtig, mal analytisch-strukturiert.

Das Ziel ist nicht, *die eine* perfekte Antwort zu finden, sondern die **Bandbreite möglicher, plausibler kognitiver Reaktionen** auf ein Thema zu verstehen, die die Simulation aufzeigt.
