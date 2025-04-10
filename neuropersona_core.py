# -*- coding: utf-8 -*-
# Filename: neuropersona_core.py
"""
NeuroPersona Core Simulation Engine

Dieses Modul enthält die Kernlogik für die NeuroPersona-Simulation,
gekapselt in einer Funktion `run_neuropersona_simulation`, die von
einem Orchestrator aufgerufen werden kann. Es enthält auch eine optionale,
vereinfachte GUI zum Starten des Workflows und zum Einstellen von Parametern.
"""

import pandas as pd
import numpy as np
import random
import matplotlib
matplotlib.use('TkAgg')  # TkAgg Backend für Kompatibilität mit Tkinter
import matplotlib.pyplot as plt
from tqdm import tqdm
import tkinter as tk
from tkinter import ttk, messagebox, filedialog, scrolledtext
import os
import time
import threading
from collections import Counter
import json
import importlib # Für dynamischen Import in GUI

# --- Konstanten ---
MODEL_FILENAME = "neuropersona_final_state.json" # Optional: Speichern des Endzustands
SETTINGS_FILENAME = "neuropersona_gui_settings.json" # Für GUI Parameter (LR, DR, Epochs)
PLOTS_FOLDER = "plots" # Ordner für Plots
DEFAULT_EPOCHS = 20
DEFAULT_LEARNING_RATE = 0.1
DEFAULT_DECAY_RATE = 0.01
DEFAULT_REWARD_INTERVAL = 5
DEFAULT_ACTIVATION_THRESHOLD_PROMOTION = 0.7
DEFAULT_HISTORY_LENGTH_MAP_PROMOTION = {"short_term": 5, "mid_term": 20}
DEFAULT_MODULE_CATEGORY_WEIGHT = 0.15 # Standardgewicht Modul -> Kategorie

# --- Neuronentyp-Funktion ---
def random_neuron_type() -> str:
    """Wählt zufällig einen Neuronentyp basierend auf biologischen Wahrscheinlichkeiten."""
    r = random.random()
    if r < 0.7: return "excitatory"
    elif r < 0.95: return "inhibitory"
    else: return "interneuron"

# --- Hilfsfunktionen ---
def sigmoid(x):
    """Sigmoid-Aktivierungsfunktion mit Clipping zur Vermeidung von Overflows."""
    with np.errstate(over='ignore', under='ignore'): # Ignoriere Overflow/Underflow Warnungen
        # Clip x, um extreme Werte zu vermeiden, die zu NaN oder Inf führen könnten
        result = 1 / (1 + np.exp(-np.clip(x, -700, 700)))
    return result

def add_activation_noise(activation, noise_level=0.05):
    """Fügt der Aktivierung gaußsches Rauschen hinzu."""
    noise = np.random.normal(0, noise_level)
    return np.clip(activation + noise, 0.0, 1.0) # Stelle sicher, dass Aktivierung im Bereich [0, 1] bleibt

# --- HTML-Report Funktion ---
def create_html_report(final_summary: str, final_recommendation: str, interpretation_log: list, important_categories: list, plots_folder: str = PLOTS_FOLDER, output_html: str = "neuropersona_report.html") -> None:
    """Erstellt einen HTML-Bericht der Simulationsergebnisse."""
    if not os.path.exists(plots_folder):
        print(f"Info: Plot-Ordner '{plots_folder}' wird erstellt.")
        os.makedirs(plots_folder, exist_ok=True)

    try:
        plots = sorted([f for f in os.listdir(plots_folder) if f.endswith(".png")])
    except FileNotFoundError:
        plots = []
        print(f"Warnung: Plot-Ordner '{plots_folder}' nicht gefunden beim Erstellen des HTML-Reports.")

    recommendation_color = {
        "Empfehlung": "#28a745", "Abwarten": "#ffc107", "Abraten": "#dc3545"
    }.get(final_recommendation, "#6c757d") # Standardfarbe Grau

    try:
        with open(output_html, "w", encoding="utf-8") as f:
            f.write("""<!DOCTYPE html><html lang='de'><head><meta charset='UTF-8'><title>NeuroPersona Analysebericht</title>
    <style>body{font-family:Arial,sans-serif;margin:30px;background-color:#f8f9fa;color:#212529}h1,h2{color:#343a40}.prognosis{background:""" + recommendation_color + """;color:white;padding:20px;border-radius:8px;font-size:1.2em}details{margin-top:20px;background:#fff;border:1px solid #dee2e6;border-radius:8px;padding:15px;box-shadow:0 2px 6px rgba(0,0,0,.05)}summary{font-weight:700;font-size:1.1em;cursor:pointer}img{max-width:80%;height:auto;margin-top:10px;border:1px solid #dee2e6;border-radius:5px;display:block;margin-left:auto; margin-right:auto;}.footer{margin-top:50px;text-align:center;font-size:.9em;color:#adb5bd}</style></head><body>""")
            f.write(f"<h1>NeuroPersona Analysebericht</h1>")
            f.write(f"<div class='prognosis'>📈 Regelbasierte Einschätzung: <b>{final_recommendation}</b><br><br>")
            if important_categories:
                 f.write("Wichtigkeit der Top-Kategorien:<br>")
                 for cat, importance in important_categories[:5]: # Zeige nur Top 5
                     f.write(f"🔹 <b>{cat}</b>: {importance}<br>")
            else:
                f.write("Keine wichtigen Kategorien identifiziert.")
            f.write("</div>")
            f.write("<details open><summary>📋 Zusammenfassung des Netzwerkzustands</summary>")
            f.write(f"<pre style='white-space: pre-wrap; word-wrap: break-word;'>{str(final_summary)}</pre></details>")

            if interpretation_log:
                f.write("<details><summary>📈 Analyseverlauf (Epochen)</summary>")
                for entry in interpretation_log:
                    epoch = entry.get('epoch','-')
                    dominant = entry.get('dominant_category','-')
                    activation = entry.get('dominant_activation', 0.0)
                    # Stelle sicher, dass Aktivierung ein Float ist, bevor formatiert wird
                    try:
                        activation_val = float(activation)
                        activation_str = f"{activation_val:.2f}" if not np.isnan(activation_val) else "NaN"
                    except (ValueError, TypeError):
                        activation_str = "N/A"
                    f.write(f"<p><b>Epoche {epoch}:</b> Dominant: {dominant} (Aktivierung: {activation_str})</p>")
                f.write("</details>")

            if plots:
                f.write("<details open><summary>🖼️ Visualisierungen</summary>")
                for plot in plots:
                    # Verwende relative Pfade, damit es im Browser funktioniert
                    f.write(f"<img src='{plots_folder}/{plot}' alt='{plot}'><br>")
                f.write("</details>")
            else:
                 f.write("<details><summary>🖼️ Visualisierungen</summary><p>Keine Plots gefunden/generiert.</p></details>")

            f.write("<div class='footer'>Erstellt mit NeuroPersona KI-System</div></body></html>")
        print(f"✅ HTML-Report erstellt: {output_html}")
    except IOError as e:
        print(f"FEHLER beim Schreiben des HTML-Reports '{output_html}': {e}")
    except Exception as e:
        print(f"Unbekannter FEHLER beim Erstellen des HTML-Reports: {e}")


# --- Netzwerk-Hilfsfunktionen ---
def decay_weights(nodes_list, decay_rate=0.002, forgetting_curve=0.98):
    """Reduziert alle Gewichte im Netzwerk leicht."""
    factor = np.clip((1 - decay_rate) * forgetting_curve, 0.0, 1.0) # Sicherstellen > 0
    for node in nodes_list:
        if hasattr(node, 'connections'):
            for conn in node.connections:
                conn.weight *= factor

def reward_connections(nodes_list, target_label, reward_factor=0.05):
    """Stärkt eingehende Verbindungen zu einem Zielknoten."""
    for node in nodes_list:
         if hasattr(node, 'connections'):
            for conn in node.connections:
                # Belohne Verbindung, WENN das Ziel die target_label ist
                if hasattr(conn.target_node, 'label') and conn.target_node.label == target_label:
                    # Erhöhe Gewicht proportional zur Aktivierung des *Quellknotens*
                    conn.weight += reward_factor * getattr(node, 'activation', 0.0)
                    conn.weight = np.clip(conn.weight, 0.0, 1.0)

def apply_emotion_weight(activation, node_label, emotion_weights, emotional_state=1.0):
    """Modifiziert eine Aktivierung basierend auf Emotion."""
    emotion_factor = emotion_weights.get(node_label, 1.0) * emotional_state
    return np.clip(activation * emotion_factor, 0.0, 1.0)

# --- Text/Numerisch Konvertierung und Vorverarbeitung ---
def convert_text_answer_to_numeric(answer_text):
    """Konvertiert vordefinierte Textantworten in numerische Werte (0-1)."""
    if not isinstance(answer_text, str):
        answer_text = str(answer_text) # Versuch der Konvertierung zu String
    answer_text = answer_text.strip().lower()
    if not answer_text: return 0.5 # Leerer String -> neutral

    # Erweiterte und robustere Mappings
    mapping = {
        # Häufigkeit
        "täglich": 0.95, "mehrmals wöchentlich": 0.85, "wöchentlich": 0.75,
        "mehrmals monatlich": 0.65, "monatlich": 0.55, "quartalsweise": 0.45,
        "halbjährlich": 0.40, "jährlich": 0.38, "selten": 0.35, "sehr selten": 0.20, "nie": 0.10,
        # Wichtigkeit/Zustimmung
        "sehr wichtig": 0.95, "wichtig": 0.80, "eher wichtig": 0.65, "neutral": 0.50, "vielleicht": 0.50,
        "eher unwichtig": 0.35, "unwichtig": 0.20, "sehr unwichtig": 0.05, "gar nicht wichtig": 0.05,
        "stimme voll zu": 0.95, "stimme zu": 0.80, "stimme eher zu": 0.65,
        "unentschieden": 0.50, "stimme eher nicht zu": 0.35, "stimme nicht zu": 0.20,
        "lehne voll ab": 0.05, "lehne ab": 0.05, "stimme überhaupt nicht zu": 0.05,
        # Zufriedenheit
        "sehr zufrieden": 0.95, "zufrieden": 0.80, "eher zufrieden": 0.65,
        "mittelmäßig": 0.50, "ok": 0.50, "geht so": 0.45,
        "eher unzufrieden": 0.35, "unzufrieden": 0.20, "sehr unzufrieden": 0.05,
        # Ja/Nein und Skalen
        "ja": 0.90, "eher ja": 0.70, "eher nein": 0.30, "nein": 0.10,
        "hoch": 0.90, "mittel": 0.50, "gering": 0.10, "sehr hoch": 0.95, "sehr gering": 0.05,
        "positiv": 0.85, "negativ": 0.15, "stark positiv": 0.95, "stark negativ": 0.05, "neutral": 0.5,
        # Tendenzen
        "steigend": 0.8, "fallend": 0.2, "stabil": 0.5, "gleichbleibend": 0.5, "stark steigend": 0.95, "stark fallend": 0.05,
        # Wahrscheinlichkeit
         "sehr wahrscheinlich": 0.95, "wahrscheinlich": 0.80, "eher wahrscheinlich": 0.65,
         "unwahrscheinlich": 0.20, "sehr unwahrscheinlich": 0.05, "möglich": 0.55,
        # Andere
        "gut": 0.8, "schlecht": 0.2, "sehr gut": 0.95, "sehr schlecht": 0.05,
        "unbekannt": 0.5, "nicht anwendbar": 0.5, "keine angabe": 0.5,
        "viel": 0.85, "wenig": 0.15, "keine": 0.05, "alle": 0.95,
    }
    # Exakter Match zuerst
    value = mapping.get(answer_text)
    if value is not None:
        return value

    # Teilstring-Match (vorsichtig verwenden!)
    if "steigend" in answer_text: return 0.7
    if "fallend" in answer_text: return 0.3
    if "positiv" in answer_text: return 0.75
    if "negativ" in answer_text: return 0.25
    if "zufrieden" in answer_text: return 0.7

    # Versuch, Zahlen direkt zu parsen
    try:
        if " von " in answer_text:
            parts = answer_text.split(" von ")
            num = float(parts[0].strip().replace(',', '.'))
            den = float(parts[1].strip().replace(',', '.'))
            return np.clip(num / den, 0.0, 1.0) if den != 0 else 0.5
        elif "%" in answer_text:
            num_str = answer_text.replace("%", "").strip().replace(',', '.')
            num = float(num_str)
            return np.clip(num / 100.0, 0.0, 1.0)
        else:
            num = float(answer_text.replace(',', '.'))
            if 0 <= num <= 1: return num
            elif 1 < num <= 5: return np.clip((num - 1) / 4.0, 0.0, 1.0)
            elif 5 < num <= 10: return np.clip((num - 1) / 9.0, 0.0, 1.0)
    except (ValueError, TypeError, IndexError):
        pass

    return 0.5 # Fallback

def preprocess_data(input_data: pd.DataFrame) -> pd.DataFrame:
    """Verarbeitet DataFrame für normalisierte Antworten (zeilenweise)."""
    print("Starte Datenvorverarbeitung...")
    if not isinstance(input_data, pd.DataFrame):
        print("FEHLER: Input ist kein Pandas DataFrame.")
        return pd.DataFrame(columns=['Frage', 'Antwort', 'Kategorie', 'normalized_answer'])

    required_cols = ['Frage', 'Antwort', 'Kategorie']
    if not all(col in input_data.columns for col in required_cols):
        missing = [col for col in required_cols if col not in input_data.columns]
        print(f"FEHLER: Input DataFrame fehlen die Spalten: {missing}.")
        return pd.DataFrame(columns=required_cols + ['normalized_answer'])

    data = input_data.copy()
    data['Frage'] = data['Frage'].astype(str)
    data['Kategorie'] = data['Kategorie'].astype(str).str.strip()
    data['normalized_answer'] = data['Antwort'].apply(convert_text_answer_to_numeric)

    if data['Kategorie'].isnull().any() or (data['Kategorie'] == '').any():
        print("Warnung: Einige Kategorien sind leer oder NaN. Werden als 'Unbekannt' behandelt.")
        data['Kategorie'].fillna('Unbekannt', inplace=True)
        data['Kategorie'].replace('', 'Unbekannt', inplace=True)

    print(f"Datenvorverarbeitung abgeschlossen. {len(data)} Zeilen verarbeitet.")
    return data


# --- Netzwerk-Hilfsfunktionen (Rest) ---
def social_influence(nodes_list, social_network, influence_factor=0.05):
    """Simuliert sozialen Einfluss basierend auf Aktivierungen."""
    for node in nodes_list:
        if not hasattr(node, 'label') or node.label not in social_network: continue
        social_impact_value = social_network.get(node.label, 0.0)
        if social_impact_value <= 0: continue
        for source_node in nodes_list:
            if hasattr(source_node, 'connections'):
                for conn in source_node.connections:
                    if hasattr(conn.target_node, 'label') and conn.target_node.label == node.label:
                        conn.weight += social_impact_value * influence_factor * getattr(source_node, 'activation', 0.5)
                        conn.weight = np.clip(conn.weight, 0, 1.0)

def update_emotional_state(emotional_state, base_activation_level=0.5, change_rate=0.02, volatility=0.05):
    """Passt den globalen emotionalen Zustand leicht an."""
    change = (base_activation_level - emotional_state) * change_rate
    noise = np.random.normal(0, volatility)
    emotional_state += change + noise
    return np.clip(emotional_state, 0.7, 1.5)

def apply_contextual_factors(activation, node, context_factors):
    """Modifiziert Aktivierung basierend auf externen Kontextfaktoren."""
    if not hasattr(node, 'label') or not context_factors: return activation
    context_factor = context_factors.get(node.label, 1.0) * random.uniform(0.95, 1.05)
    return np.clip(activation * context_factor, 0.0, 1.0)

def hebbian_learning(node, learning_rate=0.1, weight_limit=1.0, reg_factor=0.001):
    """Stärkt Verbindungen zwischen gleichzeitig aktiven Knoten (Hebb'sche Regel)."""
    if not hasattr(node, 'connections') or not hasattr(node, 'activation'): return
    node_activation = getattr(node, 'activation', 0.0)
    if node_activation < 0.1: return
    for connection in node.connections:
        target_activation = getattr(connection.target_node, 'activation', 0.0)
        if target_activation < 0.1: continue
        delta_weight = learning_rate * node_activation * target_activation
        connection.weight += delta_weight
        connection.weight -= reg_factor * connection.weight
        connection.weight = np.clip(connection.weight, 0.0, weight_limit)

# --- Klassen für Netzwerkstruktur ---
class Connection:
    """Repräsentiert eine gerichtete Verbindung zwischen zwei Knoten."""
    def __init__(self, target_node, weight=None):
        self.target_node = target_node
        self.weight = weight if weight is not None else random.uniform(0.05, 0.3)

class Node:
    """Basisklasse für einen Knoten im Netzwerk."""
    def __init__(self, label: str, neuron_type: str = "excitatory"):
        self.label = label
        self.neuron_type = neuron_type
        self.connections = []
        self.activation = 0.0
        self.activation_sum = 0.0
        self.activation_history = []

    def add_connection(self, target_node, weight=None):
        """Fügt eine ausgehende Verbindung hinzu, falls sie noch nicht existiert."""
        if target_node is self: return
        if not any(conn.target_node == target_node for conn in self.connections):
            self.connections.append(Connection(target_node, weight))

class MemoryNode(Node):
    """Ein Knoten mit Gedächtniseigenschaften (Kurz-, Mittel-, Langzeit)."""
    def __init__(self, label: str, memory_type="short_term", neuron_type="excitatory"):
        super().__init__(label, neuron_type=neuron_type)
        self.memory_type = memory_type
        self.retention_times = {"short_term": 5, "mid_term": 20, "long_term": 100}
        self.retention_time = self.retention_times.get(memory_type, 20)
        self.time_in_memory = 0
        self.history_length_maps = DEFAULT_HISTORY_LENGTH_MAP_PROMOTION

    def decay(self, decay_rate, context_factors={}, emotional_state=1.0):
        """Reduziert Gewichte ausgehender Verbindungen basierend auf Memory-Typ."""
        pass # Aktuell in globalem decay_weights

    def promote(self, activation_threshold=DEFAULT_ACTIVATION_THRESHOLD_PROMOTION, history_length_map=None):
        """Prüft, ob der Knoten zu einem stabileren Gedächtnistyp aufsteigen soll."""
        if history_length_map is None:
            history_length_map = self.history_length_maps
        required_length = history_length_map.get(self.memory_type)
        if required_length is None or self.memory_type == "long_term": return

        if not hasattr(self, 'activation_history') or len(self.activation_history) < required_length:
            return

        avg_recent_activation = np.mean(self.activation_history[-required_length:])
        if avg_recent_activation > activation_threshold:
            original_type = self.memory_type
            if self.memory_type == "short_term":
                self.memory_type = "mid_term"
                self.retention_time = self.retention_times.get("mid_term", 20)
            elif self.memory_type == "mid_term":
                self.memory_type = "long_term"
                self.retention_time = self.retention_times.get("long_term", 100)

            if original_type != self.memory_type:
                print(f"Info: Knoten '{self.label}' zu '{self.memory_type}' befördert (AvgAct: {avg_recent_activation:.2f}).")
                self.time_in_memory = 0

# --- Spezialisierte Modul-Klassen ---
class CortexCreativus(Node):
    def __init__(self, label="Cortex Creativus", neuron_type=None):
        neuron_type = neuron_type if neuron_type else random_neuron_type()
        super().__init__(label, neuron_type=neuron_type)
    def generate_new_ideas(self, active_nodes):
        ideas = []
        sorted_nodes = sorted([n for n in active_nodes if getattr(n, 'activation', 0.0) > 0.6],
                              key=lambda n: getattr(n, 'activation', 0.0), reverse=True)
        if len(sorted_nodes) >= 2:
            ideas.append(f"Idea_combining_{sorted_nodes[0].label}_and_{sorted_nodes[1].label}")
        elif len(sorted_nodes) == 1:
            ideas.append(f"Idea_inspired_by_{sorted_nodes[0].label}")
        return ideas

class SimulatrixNeuralis(Node):
    def __init__(self, label="Simulatrix Neuralis", neuron_type=None):
        neuron_type = neuron_type if neuron_type else random_neuron_type()
        super().__init__(label, neuron_type=neuron_type)
    def simulate_scenarios(self, active_nodes):
        scenarios = []
        for node in active_nodes:
            if getattr(node, 'activation', 0.0) > 0.7:
                prefix = "PositiveScenario" if node.activation > 0.8 else "NeutralScenario"
                scenarios.append(f"{prefix}_if_{node.label}_persists")
        return scenarios

class CortexCriticus(Node):
    def __init__(self, label="Cortex Criticus", neuron_type=None):
        neuron_type = neuron_type if neuron_type else "inhibitory"
        super().__init__(label, neuron_type=neuron_type)
    def evaluate_ideas(self, ideas, current_network_state_nodes):
        evaluated = []
        activations = [getattr(n, 'activation', 0.0) for n in current_network_state_nodes if hasattr(n, 'activation')]
        avg_activation = np.mean(activations) if activations else 0.5
        for idea in ideas:
            score = np.clip(random.uniform(0.0, 0.7) * avg_activation * 1.2, 0.0, 1.0)
            evaluated.append({"idea": idea, "score": round(score, 3)})
        return evaluated

class LimbusAffektus(Node):
    def __init__(self, label="Limbus Affektus", neuron_type=None):
        neuron_type = neuron_type if neuron_type else random_neuron_type()
        super().__init__(label, neuron_type=neuron_type)
    def update_emotion(self, current_emotional_state, active_nodes):
        positive_triggers = ["chance", "wachstum", "positiv", "zufrieden", "wichtig", "gut", "hoch", "ja", "steigend"]
        negative_triggers = ["risiko", "problem", "negativ", "unzufrieden", "unwichtig", "schlecht", "gering", "nein", "fallend"]
        pos_activation_sum = 0; neg_activation_sum = 0; pos_count = 0; neg_count = 0
        for node in active_nodes:
            if not hasattr(node, 'label') or not hasattr(node, 'activation'): continue
            label_lower = node.label.lower(); activation = node.activation
            if activation < 0.4: continue
            if any(trigger in label_lower for trigger in positive_triggers): pos_activation_sum += activation; pos_count += 1
            if any(trigger in label_lower for trigger in negative_triggers): neg_activation_sum += activation; neg_count += 1
        pos_activation_avg = pos_activation_sum / pos_count if pos_count > 0 else 0
        neg_activation_avg = neg_activation_sum / neg_count if neg_count > 0 else 0
        change = (pos_activation_avg - neg_activation_avg) * 0.05
        new_emotional_state = current_emotional_state + change + np.random.normal(0, 0.02)
        return np.clip(new_emotional_state, 0.7, 1.5)

class MetaCognitio(Node):
    def __init__(self, label="Meta Cognitio", neuron_type=None):
        neuron_type = neuron_type if neuron_type else random_neuron_type()
        super().__init__(label, neuron_type=neuron_type)
    def optimize_learning_parameters(self, nodes_list, current_lr, current_dr):
        activations = [getattr(n, 'activation', 0.0) for n in nodes_list if hasattr(n, 'activation')]
        avg_activation = np.mean(activations) if activations else 0.0
        std_activation = np.std(activations) if len(activations) > 1 else 0.0
        new_lr, new_dr = current_lr, current_dr
        if avg_activation > 0.75: new_lr *= 0.97; new_dr *= 1.03
        elif avg_activation < 0.35: new_lr *= 1.02
        if std_activation > 0.3: new_lr *= 0.98
        return np.clip(new_lr, 0.01, 0.5), np.clip(new_dr, 0.0005, 0.05)

class CortexSocialis(Node):
    def __init__(self, label="Cortex Socialis", neuron_type=None):
        neuron_type = neuron_type if neuron_type else random_neuron_type()
        super().__init__(label, neuron_type=neuron_type)
    def update_social_factors(self, social_network, active_nodes):
        for node in active_nodes:
             if not hasattr(node, 'label'): continue
             label = node.label; activation = getattr(node, 'activation', 0.0)
             current_factor = social_network.get(label, 0.3)
             if activation > 0.75: social_network[label] = min(current_factor * 1.05, 1.0)
             elif activation < 0.25: social_network[label] = max(current_factor * 0.95, 0.1)
        return social_network

# --- Verbindungsfunktion ---
def connect_network_components(category_nodes, module_nodes, question_nodes):
    """Verbindet die verschiedenen Teile des Netzwerks dynamisch."""
    print(f"Verbinde {len(category_nodes)} Kategorien, {len(module_nodes)} Modules, {len(question_nodes)} Fragen...")
    for i, module1 in enumerate(module_nodes):
        for j, module2 in enumerate(module_nodes):
            if i < j:
                weight = random.uniform(0.02, 0.12)
                module1.add_connection(module2, weight=weight)
                module2.add_connection(module1, weight=weight)
    for module in module_nodes:
        for cat_node in category_nodes:
            module.add_connection(cat_node, weight=DEFAULT_MODULE_CATEGORY_WEIGHT * random.uniform(0.8, 1.2))
            cat_node.add_connection(module, weight=random.uniform(0.05, 0.20))
    for i, cat1 in enumerate(category_nodes):
        for j, cat2 in enumerate(category_nodes):
            if i < j:
                 weight = random.uniform(0.01, 0.08)
                 cat1.add_connection(cat2, weight=weight)
                 cat2.add_connection(cat1, weight=weight)
    print("Netzwerkkomponenten verbunden.")
    all_nodes = category_nodes + module_nodes + question_nodes
    return all_nodes

# --- Visualisierungsfunktionen ---
def filter_module_history(activation_history, module_labels):
    """Extrahiert die Aktivierungshistorie nur für die Module."""
    return {label: activation_history[label] for label in module_labels if label in activation_history}

def plot_activation_and_weights(activation_history: dict, weights_history: dict, filename: str = "aktivierung_und_gewichte.png") -> plt.Figure | None:
    """Erstellt Plot für Aktivierungs- und Gewichtsentwicklung."""
    print("Erstelle Plot: Aktivierungs- und Gewichtsentwicklung...")
    if not activation_history and not weights_history: print("Keine Daten für Aktivierungs/Gewichtsplot."); return None
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8)); plot_count_act = 0; max_lines = 25
    sorted_act_keys = sorted(activation_history.keys(), key=lambda k: np.std(activation_history.get(k,[])) if len(activation_history.get(k,[])) > 1 else 0, reverse=True)
    for label in sorted_act_keys:
        activations = activation_history.get(label, [])
        if activations and plot_count_act < max_lines: ax1.plot(range(1, len(activations) + 1), activations, label=label, alpha=0.7); plot_count_act += 1
    ax1.set_title(f"Aktivierungsentwicklung (Top {plot_count_act} dynamisch)"); ax1.set_xlabel("Epoche"); ax1.set_ylabel("Aktivierung"); ax1.legend(fontsize='small', loc='upper right', bbox_to_anchor=(1.25, 1.02)); ax1.grid(True, alpha=0.5)
    plot_count_weights = 0
    if weights_history:
        sorted_weights_keys = sorted(weights_history.keys(), key=lambda k: np.std(weights_history.get(k,[])) if len(weights_history.get(k,[])) > 1 else 0, reverse=True)
        for label in sorted_weights_keys:
            weights = weights_history.get(label, [])
            if weights and plot_count_weights < max_lines: ax2.plot(range(1, len(weights) + 1), weights, label=label, alpha=0.6); plot_count_weights += 1
    ax2.set_title(f"Gewichtsentwicklung (Top {plot_count_weights} dynamisch)"); ax2.set_xlabel("Epoche"); ax2.set_ylabel("Gewicht")
    if plot_count_weights > 0: ax2.legend(fontsize='small', loc='upper right', bbox_to_anchor=(1.25, 1.02))
    ax2.grid(True, alpha=0.5); plt.subplots_adjust(right=0.85); plt.tight_layout(rect=[0, 0, 0.85, 1])
    try: filepath = os.path.join(PLOTS_FOLDER, filename); fig.savefig(filepath, bbox_inches='tight'); print(f"Plot gespeichert: {filepath}")
    except Exception as e: print(f"FEHLER beim Speichern von Plot '{filename}': {e}"); return fig # Return fig even on save error

def plot_dynamics(activation_history, weights_history, filename: str = "netzwerk_dynamiken.png") -> plt.Figure | None:
    """Erstellt Plot für verschiedene Netzwerkdynamiken."""
    print("Erstelle Plot: Netzwerk-Dynamiken...")
    if not activation_history and not weights_history: print("Keine Daten für Dynamikplot."); return None
    fig, axs = plt.subplots(2, 2, figsize=(16, 12)); axs = axs.flatten(); num_epochs_act = max((len(a) for a in activation_history.values()), default=0)
    avg_activations_per_epoch = []; std_activations_per_epoch = []
    if num_epochs_act > 0:
        for epoch_idx in range(num_epochs_act): epoch_activations = [history[epoch_idx] for history in activation_history.values() if len(history) > epoch_idx]; avg_activations_per_epoch.append(np.mean(epoch_activations) if epoch_activations else np.nan); std_activations_per_epoch.append(np.std(epoch_activations) if len(epoch_activations) > 1 else 0.0)
        axs[0].plot(range(1, num_epochs_act + 1), avg_activations_per_epoch, label="Avg. Aktivierung"); axs[0].fill_between(range(1, num_epochs_act + 1), np.array(avg_activations_per_epoch) - np.array(std_activations_per_epoch), np.array(avg_activations_per_epoch) + np.array(std_activations_per_epoch), alpha=0.2, label="StdAbw")
    axs[0].set_title("Netzwerkaktivierung (Durchschnitt & Streuung)"); axs[0].set_xlabel("Epoche"); axs[0].set_ylabel("Aktivierung"); axs[0].grid(True, alpha=0.5); axs[0].legend(); axs[0].set_ylim(0, 1)
    avg_weights_per_epoch = []; std_weights_per_epoch = []; num_epochs_weights = max((len(w) for w in weights_history.values()), default=0) if weights_history else 0
    if num_epochs_weights > 0:
        for epoch_idx in range(num_epochs_weights): epoch_weights = [weights[epoch_idx] for weights in weights_history.values() if len(weights) > epoch_idx]; avg_weights_per_epoch.append(np.mean(epoch_weights) if epoch_weights else np.nan); std_weights_per_epoch.append(np.std(epoch_weights) if len(epoch_weights) > 1 else 0.0)
        axs[1].plot(range(1, num_epochs_weights + 1), avg_weights_per_epoch, label="Avg. Gewicht", color='green'); axs[1].fill_between(range(1, num_epochs_weights + 1), np.array(avg_weights_per_epoch) - np.array(std_weights_per_epoch), np.array(avg_weights_per_epoch) + np.array(std_weights_per_epoch), alpha=0.2, label="StdAbw", color='green')
    axs[1].set_title("Durchschnittliche Gewichtsentwicklung"); axs[1].set_xlabel("Epoche"); axs[1].set_ylabel("Gewicht"); axs[1].grid(True, alpha=0.5); axs[1].legend(); axs[1].set_ylim(0, 1)
    active_nodes_per_epoch = []
    if num_epochs_act > 0:
        total_nodes = len(activation_history);
        for epoch_idx in range(num_epochs_act): epoch_activations = [history[epoch_idx] for history in activation_history.values() if len(history) > epoch_idx]; active_count = sum(1 for act in epoch_activations if act > 0.5); active_nodes_per_epoch.append(active_count)
        axs[2].plot(range(1, num_epochs_act + 1), active_nodes_per_epoch, label="Aktive Knoten (>0.5)", color='red'); axs[2].set_ylim(0, total_nodes + 1);
    axs[2].set_title("Netzwerk-Aktivität (Knoten > 0.5)"); axs[2].set_xlabel("Epoche"); axs[2].set_ylabel("Anzahl Knoten"); axs[2].grid(True, alpha=0.5); axs[2].legend()
    dominant_activation_per_epoch = []
    module_labels_set = {m().label for m in [CortexCreativus, SimulatrixNeuralis, CortexCriticus, LimbusAffektus, MetaCognitio, CortexSocialis]}
    category_labels = [k for k,h in activation_history.items() if isinstance(h, list) and not k.startswith("Q_") and k not in module_labels_set]
    if num_epochs_act > 0 and category_labels:
            for epoch_idx in range(num_epochs_act): max_act_epoch = 0;
            for label in category_labels: history = activation_history.get(label);
            if history and len(history) > epoch_idx: max_act_epoch = max(max_act_epoch, history[epoch_idx])
            dominant_activation_per_epoch.append(max_act_epoch)
            axs[3].plot(range(1, num_epochs_act + 1), dominant_activation_per_epoch, label="Max. Kategorie-Aktivierung", color='purple'); axs[3].set_ylim(0, 1);
    axs[3].set_title("Dominanz der stärksten Kategorie"); axs[3].set_xlabel("Epoche"); axs[3].set_ylabel("Maximale Aktivierung"); axs[3].grid(True, alpha=0.5); axs[3].legend()
    plt.tight_layout()
    try: filepath = os.path.join(PLOTS_FOLDER, filename); fig.savefig(filepath, bbox_inches='tight'); print(f"Plot gespeichert: {filepath}")
    except Exception as e: print(f"FEHLER beim Speichern von Plot '{filename}': {e}"); return fig

def plot_module_activation_comparison(module_activation_history, filename: str = "modul_vergleich.png") -> plt.Figure | None:
    """Erstellt Plot zum Vergleich der Modulaktivierungen."""
    print("Erstelle Plot: Modul-Aktivierungsvergleich...")
    if not module_activation_history or not any(module_activation_history.values()): print("Keine Daten für Modulvergleichsplot."); return None
    fig = plt.figure(figsize=(12, 7)); plotted = False
    for label, activations in module_activation_history.items():
        if activations: plt.plot(range(1, len(activations) + 1), activations, label=label, linewidth=2, alpha=0.8); plotted = True
    plt.title("Vergleich der Aktivierungen der kognitiven Module"); plt.xlabel("Epoche"); plt.ylabel("Aktivierung")
    if plotted: plt.legend(loc='upper right', bbox_to_anchor=(1.2, 1.0))
    plt.grid(True, alpha=0.5); plt.ylim(0, 1); plt.tight_layout(rect=[0, 0, 0.85, 1])
    try: filepath = os.path.join(PLOTS_FOLDER, filename); fig.savefig(filepath, bbox_inches='tight'); print(f"Plot gespeichert: {filepath}")
    except Exception as e: print(f"FEHLER beim Speichern von Plot '{filename}': {e}"); return fig


# --- Netzwerk-Initialisierung ---
def initialize_network_nodes(categories):
    """Initialisiert Kategorie- und Modulknoten."""
    if not isinstance(categories, (list, np.ndarray, pd.Series)) or len(categories) == 0:
        print("FEHLER: Ungültige oder leere Kategorienliste für Initialisierung.")
        return [], []
    print(f"Initialisiere Netzwerk mit {len(categories)} Kategorien...")
    unique_categories = [str(cat) for cat in pd.unique(categories)]
    category_nodes = [MemoryNode(c, memory_type="short_term", neuron_type="excitatory") for c in unique_categories]
    for node in category_nodes: node.activation_history = []
    module_nodes = [
        CortexCreativus(), SimulatrixNeuralis(), CortexCriticus(),
        LimbusAffektus(), MetaCognitio(), CortexSocialis()
    ]
    for node in module_nodes: node.activation_history = []
    print(f"{len(category_nodes)} Kategorieknoten und {len(module_nodes)} Modulknoten initialisiert.")
    return category_nodes, module_nodes


# --- Signalpropagation ---
def propagate_signal(node, input_signal, emotion_weights, emotional_state=1.0, context_factors=None):
    """Propagiert ein Signal von einem Knoten zu seinen verbundenen Zielen."""
    if not hasattr(node, 'activation') or not hasattr(node, 'connections'): return
    try:
        activation = float(input_signal)
        if np.isnan(activation): activation = 0.0
    except (ValueError, TypeError):
        activation = 0.0
    node.activation = np.clip(activation, 0.0, 1.0)
    node.activation = add_activation_noise(node.activation)
    for connection in node.connections:
        signal_strength = node.activation * connection.weight
        if node.neuron_type == "inhibitory":
            signal_strength *= -1.0
        target_node = connection.target_node
        if hasattr(target_node, 'activation_sum'):
            target_node.activation_sum += signal_strength
            target_node.activation_sum = np.clip(target_node.activation_sum, -10, 10)


# --- Kern-Simulationsschleife ---
def simulate_learning_cycle(data: pd.DataFrame, category_nodes: list, module_nodes: list,
                            epochs: int = DEFAULT_EPOCHS, learning_rate: float = DEFAULT_LEARNING_RATE,
                            reward_interval: int = DEFAULT_REWARD_INTERVAL, decay_rate: float = DEFAULT_DECAY_RATE,
                            initial_emotional_state: float = 1.0, context_factors: dict | None = None):
    """
    Führt den eigentlichen Lernzyklus der Simulation durch.
    Gibt Historien, Logs und finale Knoten zurück.
    """
    if 'normalized_answer' not in data.columns:
        print("FEHLER: 'normalized_answer' Spalte fehlt in den Eingabedaten für Simulation.")
        return {}, {}, [], category_nodes, module_nodes

    if context_factors is None: context_factors = {}
    core_network_nodes = category_nodes + module_nodes
    module_dict = {brain.label: brain for brain in module_nodes}
    weights_history = {}; activation_history = {}; module_outputs_log = {label: [] for label in module_dict.keys()}; interpretation_log = []

    question_nodes = []
    cat_node_map = {node.label: node for node in category_nodes if hasattr(node, 'label')}
    for idx, row in data.iterrows():
        q_label = f"Q_{idx}_{row.get('Kategorie', 'Kat?')}_{str(row.get('Frage', '?'))[:15]}"
        q_node = Node(q_label, neuron_type=random_neuron_type())
        question_nodes.append(q_node)
        category_label = row.get('Kategorie')
        category_node = cat_node_map.get(category_label)
        if category_node: q_node.add_connection(category_node, weight=0.8 * random.uniform(0.9, 1.1))
        else: print(f"Warnung: Kategorie '{category_label}' nicht gefunden für Frage Q_{idx}. Keine Verbindung erstellt.")

    all_nodes_sim = connect_network_components(category_nodes, module_nodes, question_nodes)
    # Initialisiere activation_history für *alle* Knoten im Netzwerk
    activation_history = {node.label: [] for node in all_nodes_sim if hasattr(node, 'label')}

    for node in all_nodes_sim:
        if not hasattr(node, 'activation_history'): node.activation_history = []
        if not hasattr(node, 'activation_sum'): node.activation_sum = 0.0
        if not hasattr(node, 'neuron_type'): node.neuron_type = random_neuron_type()
        if hasattr(node, 'connections'):
            for conn in node.connections:
                source_label = getattr(node, 'label', None); target_label = getattr(conn.target_node, 'label', None)
                if source_label and target_label:
                    history_key = f"{source_label} → {target_label}"
                    if history_key not in weights_history: weights_history[history_key] = []

    emotion_weights = {node.label: 1.0 for node in all_nodes_sim if hasattr(node, 'label')}
    social_network = {node.label: random.uniform(0.1, 0.5) for node in category_nodes if hasattr(node, 'label')}
    current_emotional_state = initial_emotional_state; current_lr = learning_rate; current_dr = decay_rate

    for epoch in tqdm(range(epochs), desc="Simulating Learning Cycle"):
        for node in all_nodes_sim: node.activation_sum = 0.0
        for idx, q_node in enumerate(question_nodes):
            norm_answer = data['normalized_answer'].iloc[idx] if idx < len(data) else 0.5
            propagate_signal(q_node, norm_answer, emotion_weights, current_emotional_state, context_factors)
        active_category_nodes_this_epoch = []
        for node in all_nodes_sim: # Update ALL nodes (incl. Q-nodes for history)
            if node not in question_nodes: # Q-Nodes haben feste Aktivierung aus Input
                 node.activation = sigmoid(node.activation_sum)
                 node.activation = apply_emotion_weight(node.activation, getattr(node, 'label', None), emotion_weights, current_emotional_state)
                 if context_factors: node.activation = apply_contextual_factors(node.activation, node, context_factors)
                 node.activation = add_activation_noise(node.activation)
                 node.activation = np.clip(node.activation, 0.0, 1.0)

            if hasattr(node, 'label') and node.label in activation_history:
                activation_history[node.label].append(node.activation)
            elif hasattr(node, 'label'): # Fallback
                activation_history[node.label] = [node.activation]

            if node in category_nodes and node.activation > 0.5: active_category_nodes_this_epoch.append(node)
            if isinstance(node, MemoryNode):
                 if hasattr(node, 'activation_history'): node.activation_history.append(node.activation)
                 node.promote()

        if "Cortex Creativus" in module_dict: ideas = module_dict["Cortex Creativus"].generate_new_ideas(active_category_nodes_this_epoch); module_outputs_log["Cortex Creativus"].append(ideas)
        if "Simulatrix Neuralis" in module_dict: scenarios = module_dict["Simulatrix Neuralis"].simulate_scenarios(active_category_nodes_this_epoch); module_outputs_log["Simulatrix Neuralis"].append(scenarios)
        if "Cortex Criticus" in module_dict and module_outputs_log.get("Cortex Creativus") and module_outputs_log["Cortex Creativus"][-1]: evaluated = module_dict["Cortex Criticus"].evaluate_ideas(module_outputs_log["Cortex Creativus"][-1], category_nodes); module_outputs_log["Cortex Criticus"].append(evaluated)
        if "Limbus Affektus" in module_dict: current_emotional_state = module_dict["Limbus Affektus"].update_emotion(current_emotional_state, category_nodes + module_nodes); module_outputs_log["Limbus Affektus"].append(f"State: {current_emotional_state:.3f}")
        if "Meta Cognitio" in module_dict: current_lr, current_dr = module_dict["Meta Cognitio"].optimize_learning_parameters(core_network_nodes, current_lr, current_dr); module_outputs_log["Meta Cognitio"].append(f"LR:{current_lr:.4f},DR:{current_dr:.5f}")
        if "Cortex Socialis" in module_dict: social_network = module_dict["Cortex Socialis"].update_social_factors(social_network, category_nodes); module_outputs_log["Cortex Socialis"].append("Factors updated"); social_influence(all_nodes_sim, social_network)

        for node in all_nodes_sim: hebbian_learning(node, current_lr, reg_factor=0.001)
        if (epoch + 1) % reward_interval == 0 and category_nodes:
            valid_cat_labels = [n.label for n in category_nodes if hasattr(n,'label')]
            if valid_cat_labels:
                 target_category_label = random.choice(valid_cat_labels)
                 reward_connections(all_nodes_sim, target_category_label, reward_factor=0.05)
        decay_weights(all_nodes_sim, current_dr)
        for node in all_nodes_sim:
            if hasattr(node, 'connections'):
                for conn in node.connections:
                    source_label = getattr(node, 'label', None); target_label = getattr(conn.target_node, 'label', None)
                    if source_label and target_label:
                        history_key = f"{source_label} → {target_label}"
                        if history_key in weights_history: weights_history[history_key].append(conn.weight)

        epoch_interpretation = interpret_epoch_state(epoch, category_nodes, module_nodes, module_outputs_log, activation_history)
        interpretation_log.append(epoch_interpretation)

    print(f"Simulationszyklus über {epochs} Epochen abgeschlossen.")
    final_category_nodes = [n for n in all_nodes_sim if isinstance(n, MemoryNode)]
    final_module_nodes = [n for n in all_nodes_sim if isinstance(n, (CortexCreativus, SimulatrixNeuralis, CortexCriticus, LimbusAffektus, MetaCognitio, CortexSocialis))]
    return activation_history, weights_history, interpretation_log, final_category_nodes, final_module_nodes

# --- Interpretation & Report ---
def interpret_epoch_state(epoch: int, category_nodes: list, module_nodes: list, module_outputs: dict, activation_history: dict) -> dict:
    """Interpretiert den Zustand des Netzwerks in einer einzelnen Epoche."""
    interpretation = {'epoch': epoch + 1}
    valid_category_nodes = [n for n in category_nodes if hasattr(n, 'activation')]
    sorted_cats = sorted(valid_category_nodes, key=lambda n: n.activation, reverse=True)
    if sorted_cats:
        interpretation['dominant_category'] = getattr(sorted_cats[0], 'label', 'Unbekannt')
        interpretation['dominant_activation'] = round(getattr(sorted_cats[0], 'activation', 0.0), 4)
        interpretation['category_ranking'] = [(getattr(n, 'label', 'Unbekannt'), round(getattr(n, 'activation', 0.0), 4)) for n in sorted_cats[:5]]
    else:
        interpretation['dominant_category'] = 'N/A'; interpretation['dominant_activation'] = 0.0; interpretation['category_ranking'] = []
    interpretation['module_activations'] = {
        getattr(m, 'label', f'Unbekanntes_Modul_{i}'): round(getattr(m, 'activation', 0.0), 4)
        for i, m in enumerate(module_nodes) if hasattr(m, 'label')} # Sicherstellen, dass Modul Label hat
    return interpretation

def generate_final_report(category_nodes: list, module_nodes: list, original_data: pd.DataFrame, interpretation_log: list) -> tuple[str, dict]:
    """Generiert den finalen Textbericht UND ein strukturiertes Ergebnis-Dictionary."""
    print("\n--- Generiere finalen NeuroPersona Analysebericht ---")
    report_lines = ["**NeuroPersona Analysebericht**\n"]
    structured_results = {"dominant_category": "N/A", "dominant_activation": 0.0, "category_ranking": [], "module_activations": {}, "final_assessment": "Keine klare Tendenz.", "frequent_dominant_category": None}
    threshold_high = 0.65; threshold_low = 0.35
    report_lines.append("**Finale Netzwerk-Tendenzen (Kategorien):**")
    valid_category_nodes = [n for n in category_nodes if hasattr(n, 'activation') and hasattr(n, 'label')]
    sorted_categories = sorted(valid_category_nodes, key=lambda n: n.activation, reverse=True)
    category_ranking_data = []
    if not sorted_categories: report_lines.append("- Keine aktiven Kategorieknoten gefunden.")
    else:
        for node in sorted_categories:
            category = node.label; activation = round(node.activation, 3)
            if activation >= threshold_high: tendency_label = "Stark Aktiviert / Hohe Relevanz"
            elif activation <= threshold_low: tendency_label = "Schwach Aktiviert / Geringe Relevanz"
            else: tendency_label = "Mittel Aktiviert / Neutrale Relevanz"
            line = f"- **{category}:** {tendency_label} (Aktivierung: {activation})"
            try:
                if 'Kategorie' in original_data.columns and 'Frage' in original_data.columns:
                     example_questions = original_data[original_data['Kategorie'] == category]['Frage']
                     if not example_questions.empty: q_example = example_questions.iloc[0]; line += f" (Bsp: '{str(q_example)[:60]}...')"
            except Exception: pass
            report_lines.append(line); category_ranking_data.append({'category': category, 'activation': activation, 'tendency': tendency_label})
        structured_results["category_ranking"] = category_ranking_data; structured_results["dominant_category"] = sorted_categories[0].label; structured_results["dominant_activation"] = round(sorted_categories[0].activation, 4)
    if interpretation_log:
        report_lines.append("\n**Verlaufseindrücke:**"); dominant_cats = [log.get('dominant_category') for log in interpretation_log if log.get('dominant_category') not in ['N/A', None]]
        if dominant_cats:
             try: most_frequent, freq_count = Counter(dominant_cats).most_common(1)[0]; report_lines.append(f"- '{most_frequent}' war am häufigsten dominant (in {freq_count} Epochen)."); structured_results["frequent_dominant_category"] = most_frequent
             except IndexError: report_lines.append("- Keine dominante Kategorie im Verlauf festgestellt.")
        else: report_lines.append("- Keine dominante Kategorie im Verlauf festgestellt.")
    report_lines.append("\n**Finaler Zustand der kognitiven Module:**"); module_activation_data = {}
    for module in module_nodes:
        label = getattr(module, 'label', 'Unbekannt'); activation = round(getattr(module, 'activation', 0.0), 3); neuron_type = getattr(module, 'neuron_type', '?'); report_lines.append(f"- {label} (Typ: {neuron_type}): {activation}"); module_activation_data[label] = activation
    structured_results["module_activations"] = module_activation_data
    final_assessment_text = "Keine klare Tendenz oder dominante Kategorie."
    if sorted_categories:
        top_cat = structured_results["dominant_category"]; top_act = structured_results["dominant_activation"]
        if top_act > threshold_high: final_assessment_text = f"Klare Tendenz bzgl. **{top_cat}**."
        elif top_act < threshold_low:
            second_cat = category_ranking_data[1]['category'] if len(category_ranking_data) > 1 else None
            if second_cat: final_assessment_text = f"Geringe Aktivierung insgesamt. '{top_cat}' und '{second_cat}' relativ am höchsten."
            else: final_assessment_text = f"Geringe Aktivierung insgesamt. Nur '{top_cat}' als schwache Tendenz."
        else: final_assessment_text = f"Keine starke Dominanz. '{top_cat}' am aktivsten, aber im mittleren Bereich."
    report_lines.append(f"\n**Gesamteinschätzung:** {final_assessment_text}"); structured_results["final_assessment"] = final_assessment_text
    final_report_text = "\n".join(report_lines); print(final_report_text); return final_report_text, structured_results

# --- Modell Speichern ---
def save_final_network_state(nodes_list, filename=MODEL_FILENAME):
    """Speichert den Zustand aller Knoten und ihrer Verbindungen."""
    model_data = {"nodes": [], "connections": []}; valid_nodes = [node for node in nodes_list if hasattr(node, 'label')]; node_labels = {node.label for node in valid_nodes}; print(f"Speichere finalen Zustand von {len(valid_nodes)} Knoten in {filename}...")
    for node in valid_nodes:
        node_info = { "label": node.label, "activation": round(getattr(node, 'activation', 0.0), 4), "neuron_type": getattr(node, 'neuron_type', "excitatory")}
        if isinstance(node, MemoryNode): node_info.update({"memory_type": node.memory_type, "time_in_memory": node.time_in_memory})
        model_data["nodes"].append(node_info)
        if hasattr(node, 'connections'):
            for conn in node.connections:
                target_label = getattr(conn.target_node, 'label', None)
                if target_label and target_label in node_labels: model_data["connections"].append({"source": node.label, "target": target_label, "weight": round(conn.weight, 4)})
    try:
        with open(filename, "w", encoding='utf-8') as file: json.dump(model_data, file, indent=2, ensure_ascii=False); print(f"Finaler Netzwerkzustand erfolgreich gespeichert in {filename}")
    except (IOError, TypeError) as e: print(f"FEHLER beim Speichern des finalen Netzwerkzustands: {e}")

# --- Plot Hilfsfunktion ---
def safe_show_plot(fig=None):
    """Zeigt Plots sicher an (hauptsächlich für interaktive GUI)."""
    if threading.current_thread() is not threading.main_thread(): return
    try:
        if fig and hasattr(fig, 'axes') and fig.axes: plt.figure(fig.number); plt.pause(0.1)
        elif plt.get_fignums(): plt.show(block=False); plt.pause(0.1)
    except Exception as e: print(f"Fehler beim Anzeigen des Plots: {e}")

def get_important_categories(category_nodes):
    """Ermittelt die Wichtigkeit von Kategorien basierend auf Aktivierung."""
    important_categories = []
    for node in category_nodes:
        if not hasattr(node, 'label') or not hasattr(node, 'activation'): continue
        act = node.activation
        if act >= 0.8: importance = "sehr hoch"
        elif act >= 0.6: importance = "hoch"
        elif act >= 0.4: importance = "mittel"
        elif act >= 0.2: importance = "gering"
        else: importance = "sehr gering"
        important_categories.append((node.label, importance))
    important_categories.sort(key=lambda item: getattr(next((n for n in category_nodes if hasattr(n,'label') and n.label == item[0]), None), 'activation', 0.0), reverse=True)
    return important_categories

# --- *** HAUPTFUNKTION FÜR EXTERNE AUFRUFE *** ---
def run_neuropersona_simulation(input_df: pd.DataFrame,
                                epochs: int = DEFAULT_EPOCHS,
                                learning_rate: float = DEFAULT_LEARNING_RATE,
                                decay_rate: float = DEFAULT_DECAY_RATE,
                                reward_interval: int = DEFAULT_REWARD_INTERVAL,
                                generate_plots: bool = True,
                                save_state: bool = False
                               ) -> tuple[str | None, dict | None]:
    """
    Führt die gesamte NeuroPersona-Simulation für gegebene Daten durch.
    (Funktionsbeschreibung wie zuvor)
    """
    start_time = time.time(); print(f"\n--- Starte NeuroPersona Simulation ---"); print(f"Parameter: Epochen={epochs}, LR={learning_rate}, DR={decay_rate}, RI={reward_interval}")
    if not isinstance(input_df, pd.DataFrame) or input_df.empty: print("FEHLER: Ungültiges oder leeres Input-DataFrame."); return None, None
    processed_data = preprocess_data(input_df)
    if processed_data.empty or 'normalized_answer' not in processed_data.columns or 'Kategorie' not in processed_data.columns: print("FEHLER: Datenvorverarbeitung fehlgeschlagen oder benötigte Spalten fehlen."); return None, None
    categories = processed_data['Kategorie'].unique()
    if len(categories) == 0: print("FEHLER: Keine eindeutigen Kategorien in den Daten gefunden nach Vorverarbeitung."); return None, None
    category_nodes, module_nodes = initialize_network_nodes(categories)
    if not category_nodes or not module_nodes: print("FEHLER: Netzwerkinitialisierung fehlgeschlagen."); return None, None
    module_labels = [m.label for m in module_nodes]
    try:
        simulation_results = simulate_learning_cycle(processed_data, category_nodes, module_nodes, epochs=epochs, learning_rate=learning_rate, reward_interval=reward_interval, decay_rate=decay_rate)
        activation_history, weights_history, interpretation_log, final_category_nodes, final_module_nodes = simulation_results
    except Exception as e: print(f"Schwerwiegender FEHLER während des Simulationszyklus: {e}"); import traceback; traceback.print_exc(); return None, None
    if not final_category_nodes: print("FEHLER: Keine finalen Kategorieknoten nach Simulation."); return None, None
    final_report_text, structured_results = generate_final_report(final_category_nodes, final_module_nodes, processed_data, interpretation_log)
    if generate_plots:
        print("\n--- Generiere Plots ---"); os.makedirs(PLOTS_FOLDER, exist_ok=True)
        try: # Zusätzlicher Try-Except für Plotting
             fig1 = plot_activation_and_weights(activation_history, weights_history)
             fig2 = plot_dynamics(activation_history, weights_history)
             module_activation_history = filter_module_history(activation_history, module_labels)
             fig3 = plot_module_activation_comparison(module_activation_history)
             plt.close('all') # Schließe alle Plot-Fenster nach dem Speichern
        except Exception as plot_error:
            print(f"FEHLER bei der Plot-Generierung: {plot_error}")
            # Simulation läuft trotzdem weiter, aber ohne Plots
    dominant_category = structured_results.get("dominant_category", "N/A"); dominant_activation = structured_results.get("dominant_activation", 0.0); final_recommendation = "Abwarten"
    if dominant_category != "N/A":
        cat_lower = dominant_category.lower()
        if ("chance" in cat_lower or "wachstum" in cat_lower or "positiv" in cat_lower or "wichtig" in cat_lower) and dominant_activation > 0.7: final_recommendation = "Empfehlung"
        elif ("risiko" in cat_lower or "problem" in cat_lower or "negativ" in cat_lower or "unwichtig" in cat_lower) and dominant_activation > 0.7: final_recommendation = "Abraten"
        elif dominant_activation < 0.4 : final_recommendation = "Abwarten"
    important_categories = get_important_categories(final_category_nodes); create_html_report(final_report_text, final_recommendation, interpretation_log, important_categories)
    if save_state: final_nodes = final_category_nodes + final_module_nodes; save_final_network_state(final_nodes, MODEL_FILENAME)
    end_time = time.time(); print(f"--- NeuroPersona Simulation abgeschlossen ({end_time - start_time:.2f}s) ---")
    return final_report_text, structured_results

# --- Optionale GUI zum Starten des Workflows ---
entry_widgets = {}

def start_gui():
    """Startet eine vereinfachte GUI zur Parametereingabe."""
    root = tk.Tk(); root.title("NeuroPersona Workflow Starter"); root.geometry("450x380"); style = ttk.Style(); style.theme_use('clam')
    main_frame = ttk.Frame(root, padding="10"); main_frame.pack(fill=tk.BOTH, expand=True)
    param_container = ttk.LabelFrame(main_frame, text="Simulationsparameter", padding="10"); param_container.pack(fill=tk.X, padx=10, pady=10); param_container.columnconfigure(1, weight=1)
    row_idx = 0
    ttk.Label(param_container, text="Lernrate:").grid(row=row_idx, column=0, sticky=tk.W, pady=3); learning_rate_entry = ttk.Entry(param_container, width=10); learning_rate_entry.insert(0, str(DEFAULT_LEARNING_RATE)); learning_rate_entry.grid(row=row_idx, column=1, sticky=tk.EW, pady=3, padx=5); entry_widgets['learning_rate'] = learning_rate_entry; row_idx+=1
    ttk.Label(param_container, text="Decay Rate:").grid(row=row_idx, column=0, sticky=tk.W, pady=3); decay_rate_entry = ttk.Entry(param_container, width=10); decay_rate_entry.insert(0, str(DEFAULT_DECAY_RATE)); decay_rate_entry.grid(row=row_idx, column=1, sticky=tk.EW, pady=3, padx=5); entry_widgets['decay_rate'] = decay_rate_entry; row_idx+=1
    ttk.Label(param_container, text="Reward Interval:").grid(row=row_idx, column=0, sticky=tk.W, pady=3); reward_interval_entry = ttk.Entry(param_container, width=10); reward_interval_entry.insert(0, str(DEFAULT_REWARD_INTERVAL)); reward_interval_entry.grid(row=row_idx, column=1, sticky=tk.EW, pady=3, padx=5); entry_widgets['reward_interval'] = reward_interval_entry; row_idx+=1
    ttk.Label(param_container, text="Epochen:").grid(row=row_idx, column=0, sticky=tk.W, pady=3); epochs_entry = ttk.Entry(param_container, width=10); epochs_entry.insert(0, str(DEFAULT_EPOCHS)); epochs_entry.grid(row=row_idx, column=1, sticky=tk.EW, pady=3, padx=5); entry_widgets['epochs'] = epochs_entry; row_idx+=1
    generate_plots_var = tk.BooleanVar(value=True); save_state_var = tk.BooleanVar(value=False)
    ttk.Checkbutton(param_container, text="Plots generieren", variable=generate_plots_var).grid(row=row_idx, column=0, columnspan=2, sticky=tk.W, pady=3); row_idx+=1
    ttk.Checkbutton(param_container, text="Finalen Zustand speichern", variable=save_state_var).grid(row=row_idx, column=0, columnspan=2, sticky=tk.W, pady=3); row_idx+=1
    prompt_container = ttk.LabelFrame(main_frame, text="Analyse-Anfrage", padding="10"); prompt_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
    ttk.Label(prompt_container, text="Geben Sie Ihre Frage oder Ihr Thema ein:").pack(anchor=tk.W, pady=(0,5)); user_prompt_text = tk.Text(prompt_container, height=4, width=40, wrap=tk.WORD, relief=tk.SOLID, borderwidth=1); user_prompt_text.pack(fill=tk.BOTH, expand=True); user_prompt_text.insert("1.0", "z.B. Wie entwickelt sich der Markt für E-Bikes 2025?")
    status_label = ttk.Label(main_frame, text="Status: Bereit", anchor=tk.W, relief=tk.SUNKEN, padding=2); status_label.pack(fill=tk.X, padx=10, pady=5)
    button_frame = ttk.Frame(main_frame); button_frame.pack(fill=tk.X, padx=10, pady=(5, 10))
    save_button = ttk.Button(button_frame, text="Params Speichern", command=lambda: save_gui_settings(status_label, generate_plots_var, save_state_var)); save_button.pack(side=tk.LEFT, padx=5)
    load_button = ttk.Button(button_frame, text="Params Laden", command=lambda: load_gui_settings(status_label, generate_plots_var, save_state_var)); load_button.pack(side=tk.LEFT, padx=5)
    start_button = ttk.Button(button_frame, text="Workflow starten", style="Accent.TButton", command=lambda: start_full_workflow_action(user_prompt_text, status_label, root, generate_plots_var, save_state_var)); start_button.pack(side=tk.RIGHT, padx=15)
    style.configure("Accent.TButton", font=("Helvetica", 10, "bold"), foreground="white", background="#007bff")

    def save_gui_settings(status_label_ref, plots_var, save_var):
        settings_data = {"basic_params": {}}
        try:
            for name, widget in entry_widgets.items(): settings_data["basic_params"][name] = widget.get()
            settings_data["options"] = {"generate_plots": plots_var.get(), "save_state": save_var.get()}
            filepath = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("JSON files", "*.json")], initialfile=SETTINGS_FILENAME, title="Simulationsparameter speichern")
            if not filepath: status_label_ref.config(text="Status: Speichern abgebrochen."); return
            with open(filepath, 'w', encoding='utf-8') as f: json.dump(settings_data, f, indent=2)
            status_label_ref.config(text=f"Status: Parameter gespeichert in {os.path.basename(filepath)}")
        except Exception as e: messagebox.showerror("Fehler Speichern", f"Konnte Parameter nicht speichern:\n{e}"); status_label_ref.config(text="Status: Fehler beim Speichern.")

    def load_gui_settings(status_label_ref, plots_var, save_var):
        try:
            filepath = filedialog.askopenfilename(defaultextension=".json", filetypes=[("JSON files", "*.json")], initialfile=SETTINGS_FILENAME, title="Simulationsparameter laden")
            if not filepath: status_label_ref.config(text="Status: Laden abgebrochen."); return
            if not os.path.exists(filepath): messagebox.showerror("Fehler", f"Datei nicht gefunden: {filepath}"); status_label_ref.config(text="Status: Ladedatei fehlt."); return
            with open(filepath, 'r', encoding='utf-8') as f: settings_data = json.load(f)
            loaded_params = settings_data.get("basic_params", {})
            for name, widget in entry_widgets.items():
                if name in loaded_params: widget.delete(0, tk.END); widget.insert(0, str(loaded_params[name]))
            loaded_options = settings_data.get("options", {})
            if "generate_plots" in loaded_options: plots_var.set(loaded_options["generate_plots"])
            if "save_state" in loaded_options: save_var.set(loaded_options["save_state"])
            status_label_ref.config(text=f"Status: Parameter geladen aus {os.path.basename(filepath)}")
        except Exception as e: messagebox.showerror("Fehler Laden", f"Konnte Parameter nicht laden:\n{e}"); status_label_ref.config(text="Status: Laden fehlgeschlagen.")

    # --- Start Action (KORRIGIERT)---
    def start_full_workflow_action(prompt_widget, status_label_ref, root_ref, plots_var, save_var):
        user_prompt = prompt_widget.get("1.0", tk.END).strip()
        if not user_prompt or user_prompt == "z.B. Wie entwickelt sich der Markt für E-Bikes 2025?":
            messagebox.showwarning("Eingabe fehlt", "Bitte geben Sie eine Analyse-Anfrage ein.")
            return

        # --- Parameter validieren ---
        try:
            lr = float(learning_rate_entry.get().replace(',', '.'))
            dr = float(decay_rate_entry.get().replace(',', '.'))
            ri = int(reward_interval_entry.get())
            ep = int(epochs_entry.get())
            gen_plots = plots_var.get()
            save_st = save_var.get()
            # Plausibilitätscheck
            if not (0 < lr <= 1.0 and 0 <= dr < 1.0 and ri >= 1 and ep >= 1):
                raise ValueError("Parameter außerhalb gültiger Bereiche (LR: 0-1, DR: 0-<1, RI/EP >= 1).")
        # --- > HIER KOMMT DER FEHLENDE except-BLOCK <---
        except ValueError as ve:
            messagebox.showerror("Eingabefehler", f"Ungültiger Parameterwert: {ve}")
            return # Verhindert Start mit ungültigen Werten
        # --- > ENDE des except-Blocks <---

        # --- Rufe Orchestrator ---
        status_label_ref.config(text="Status: Starte Workflow...")
        start_button.config(state=tk.DISABLED) # Button deaktivieren
        # Workflow in Thread starten
        threading.Thread(target=run_workflow_in_thread,
                         args=(user_prompt, lr, dr, ri, ep, gen_plots, save_st, status_label_ref, start_button, root_ref),
                         daemon=True).start()
                         
    def run_workflow_in_thread(user_prompt, lr, dr, ri, ep, gen_plots, save_st, status_label_ref, button_ref, root_ref):
        final_result_text = None
        try:
            try:
                orchestrator_module = importlib.import_module("orchestrator")
                execute_full_workflow = getattr(orchestrator_module, "execute_full_workflow")
            except (ImportError, AttributeError) as import_err:
                 print(f"FEHLER: Konnte 'execute_full_workflow' aus 'orchestrator.py' nicht laden: {import_err}")
                 if root_ref.winfo_exists():
                     root_ref.after(0, lambda: status_label_ref.config(text="Status: Fehler! Orchestrator fehlt."))
                     root_ref.after(0, lambda: messagebox.showerror("Fehler", "Orchestrator ('orchestrator.py') nicht gefunden oder Funktion fehlt."))
                 return
            def gui_status_update(message):
                 if root_ref.winfo_exists():
                     root_ref.after(0, lambda: status_label_ref.config(text=f"Status: {message}"))
            final_result_text = execute_full_workflow(user_prompt, neuropersona_epochs=ep, neuropersona_lr=lr, neuropersona_dr=dr, neuropersona_ri=ri, neuropersona_gen_plots=gen_plots, neuropersona_save_state=save_st, status_callback=gui_status_update)
            final_status = "Status: Workflow abgeschlossen."
            if final_result_text and isinstance(final_result_text, str) and ("FEHLER" in final_result_text or "Error" in final_result_text):
                 final_status = "Status: Workflow mit Fehlern beendet."
            if root_ref.winfo_exists(): root_ref.after(0, lambda: status_label_ref.config(text=final_status))
            if final_result_text and isinstance(final_result_text, str):
                if root_ref.winfo_exists(): root_ref.after(10, lambda: display_final_result(final_result_text, root_ref))
        except Exception as e:
            print(f"FEHLER im Workflow-Thread: {e}")
            import traceback; traceback.print_exc()
            if root_ref.winfo_exists():
                error_message = f"Ein unerwarteter Fehler ist aufgetreten:\n{e}"
                root_ref.after(0, lambda msg=error_message: status_label_ref.config(text="Status: Kritischer Fehler im Workflow!"))
                root_ref.after(0, lambda msg=error_message: messagebox.showerror("Workflow Fehler", msg))
        finally:
            if root_ref.winfo_exists(): root_ref.after(0, lambda: button_ref.config(state=tk.NORMAL))

    def display_final_result(result_text, parent_root):
        result_window = tk.Toplevel(parent_root); result_window.title("Workflow Ergebnis"); result_window.geometry("700x500")
        st_widget = scrolledtext.ScrolledText(result_window, wrap=tk.WORD, padx=10, pady=10, relief=tk.FLAT); st_widget.pack(fill=tk.BOTH, expand=True)
        st_widget.insert(tk.END, result_text); st_widget.configure(state='disabled')
        close_button = ttk.Button(result_window, text="Schließen", command=result_window.destroy); close_button.pack(pady=10)
        result_window.lift(); result_window.focus_force()

    if os.path.exists(SETTINGS_FILENAME): load_gui_settings(status_label, generate_plots_var, save_state_var)
    root.mainloop()

# --- Hauptprogramm-Einstieg ---
if __name__ == "__main__":
    print("Starte NeuroPersona Workflow Starter GUI...")
    if not os.path.exists(PLOTS_FOLDER): os.makedirs(PLOTS_FOLDER)
    start_gui()