# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import random
import matplotlib
matplotlib.use('TkAgg')  # Force TkAgg backend (or ensure it's used)
import matplotlib.pyplot as plt
from tqdm import tqdm  # Für Fortschrittsbalken
import tkinter as tk
from tkinter import ttk
from tkinter import messagebox # <--- Importiere messagebox explizit
from tkinter import filedialog # <--- NEU: Für Speichern/Laden Dialoge
import tkinter.scrolledtext as scrolledtext # ### NEU ### Für Gemini Output
import seaborn as sns
import networkx as nx
import json
import os
import time
import threading
from collections import Counter
import itertools # Für Grid Search

# ### NEU: Import für Gemini API ###
try:
    import google.generativeai as genai
    gemini_available = True
except ImportError:
    gemini_available = False
    print("WARNUNG: 'google-generativeai' nicht installiert. Gemini-Integration ist deaktiviert.")
    print("Installieren Sie es mit: pip install google-generativeai")

# --- Konstanten ---
MODEL_FILENAME = "neuro_persona_model.json"
SETTINGS_FILENAME = "gui_settings.json"
# ### NEU: Dateiname für Gemini-Report ###
GEMINI_REPORT_FILENAME = "gemini_analyse_bericht.txt"
DEFAULT_EPOCHS = 20
DEFAULT_LEARNING_RATE = 0.1 # ### ANGEPASST: Sicherere Standard-Lernrate ###
DEFAULT_DECAY_RATE = 0.01  # ### ANGEPASST: Etwas höherer Decay ###
DEFAULT_REWARD_INTERVAL = 5
DEFAULT_ACTIVATION_THRESHOLD_PROMOTION = 0.7
DEFAULT_HISTORY_LENGTH_MAP_PROMOTION = {"short_term": 5, "mid_term": 20}
DEFAULT_MODULE_CATEGORY_WEIGHT = 0.15

# --- Neuronentyp-Funktion ---
def random_neuron_type() -> str:
    """Wählt zufällig einen Neuronentyp basierend auf biologischen Wahrscheinlichkeiten."""
    r = random.random()
    if r < 0.7: return "excitatory"
    elif r < 0.95: return "inhibitory"
    else: return "interneuron"

# --- Debugging-Funktion ---
def debug_connections(nodes_list): pass

# --- Hilfsfunktionen ---
def sigmoid(x):
    with np.errstate(over='ignore', under='ignore'):
        result = 1 / (1 + np.exp(-np.clip(x, -700, 700)))
    return result

def add_activation_noise(activation, noise_level=0.05):
    noise = np.random.normal(0, noise_level)
    return np.clip(activation + noise, 0.0, 1.0)

# --- HTML-Report Funktion ---
def create_html_report(final_summary: str, final_recommendation: str, interpretation_log: list, important_categories: list, plots_folder: str = "plots", output_html: str = "marktanalyse_pro.html") -> None:
    # ... (Rest der Funktion bleibt unverändert) ...
    import os

    if not os.path.exists(plots_folder):
        print(f"Fehler: Plot-Ordner '{plots_folder}' existiert nicht.")
        os.makedirs(plots_folder, exist_ok=True)
        print(f"Plot-Ordner '{plots_folder}' erstellt.")

    plots = sorted([f for f in os.listdir(plots_folder) if f.endswith(".png")])

    recommendation_color = {
        "Empfehlung": "#28a745", "Abwarten": "#ffc107", "Abraten": "#dc3545"
    }.get(final_recommendation, "#6c757d")

    with open(output_html, "w", encoding="utf-8") as f:
        f.write("""<!DOCTYPE html><html lang='de'><head><meta charset='UTF-8'><title>NeuroPersona Marktanalyse 2025</title>
<style>body{font-family:Arial,sans-serif;margin:30px;background-color:#f8f9fa;color:#212529}h1,h2{color:#343a40}.prognosis{background:""" + recommendation_color + """;color:white;padding:20px;border-radius:8px;font-size:1.2em}details{margin-top:20px;background:#fff;border:1px solid #dee2e6;border-radius:8px;padding:15px;box-shadow:0 2px 6px rgba(0,0,0,.05)}summary{font-weight:700;font-size:1.1em;cursor:pointer}img{max-width:100%;height:auto;margin-top:10px;border:1px solid #dee2e6;border-radius:5px;display:block}.footer{margin-top:50px;text-align:center;font-size:.9em;color:#adb5bd}</style></head><body>""")
        f.write(f"<div class='prognosis'>📈 Prognose: <b>{final_recommendation}</b><br><br>")
        for cat, importance in important_categories:
            f.write(f"🔹 Wichtigkeit <b>{cat}</b>: {importance}<br>")
        f.write("</div>")
        f.write("<details open><summary>📋 Zusammenfassung</summary>")
        # Sicherstellen, dass final_summary ein String ist und <pre> für Formatierung nutzen
        f.write(f"<pre style='white-space: pre-wrap; word-wrap: break-word;'>{str(final_summary)}</pre></details>")
        f.write("<details><summary>📈 Analyseverlauf</summary>")
        for entry in interpretation_log:
            epoch, dominant, activation = entry.get('epoch','-'), entry.get('dominant_category','-'), entry.get('dominant_activation',0.0)
            activation_str = f"{activation:.2f}" if not np.isnan(activation) else "NaN"
            f.write(f"<p><b>Epoche {epoch}:</b> Dominant: {dominant} (Aktivierung: {activation_str})</p>")
        f.write("</details>")
        f.write("<details open><summary>🖼️ Visualisierungen</summary>")
        if plots:
            for plot in plots: f.write(f"<img src='{plots_folder}/{plot}' alt='{plot}'><br>")
        else: f.write("<p>Keine Visualisierungen gefunden.</p>")
        f.write("</details>")
        f.write("<div class='footer'>Erstellt mit NeuroPersona KI-System</div></body></html>")
    print(f"✅ Neuer professioneller HTML-Report erstellt: {output_html}")


# --- Netzwerk-Hilfsfunktionen ---
# ... (decay_weights, reward_connections, apply_emotion_weight unverändert) ...
def decay_weights(nodes_list, decay_rate=0.002, forgetting_curve=0.98):
    factor = (1 - decay_rate) * forgetting_curve
    for node in nodes_list:
        if hasattr(node, 'connections'):
            for conn in node.connections:
                conn.weight *= factor

def reward_connections(nodes_list, target_label, reward_factor=0.05):
    for node in nodes_list:
         if hasattr(node, 'connections'):
            for conn in node.connections:
                if hasattr(conn.target_node, 'label') and conn.target_node.label == target_label:
                    conn.weight += reward_factor * getattr(node, 'activation', 0.0) # Sicherer Zugriff
                    conn.weight = np.clip(conn.weight, 0, 1.0)

def apply_emotion_weight(activation, node_label, emotion_weights, emotional_state=1.0):
    emotion_factor = emotion_weights.get(node_label, 1.0) * emotional_state
    return np.clip(activation * emotion_factor, 0.0, 1.0)

# --- Text/Numerisch Konvertierung und Vorverarbeitung ---
# ... (convert_text_answer_to_numeric, preprocess_data unverändert) ...
def convert_text_answer_to_numeric(answer_text):
    """Konvertiert vordefinierte Textantworten in numerische Werte (0-1)."""
    answer_text = str(answer_text).strip().lower()
    mapping = {
        "täglich": 0.95, "wöchentlich": 0.75, "monatlich": 0.55, "selten": 0.35, "nie": 0.10,
        "sehr wichtig": 0.90, "wichtig": 0.70, "vielleicht": 0.50, "unwichtig": 0.30, "sehr unwichtig": 0.10,
        "sehr zufrieden": 0.90, "zufrieden": 0.70, "unzufrieden": 0.30,
        "ja": 0.85, "nein": 0.15
    }
    return mapping.get(answer_text)

def preprocess_data(input_data: pd.DataFrame) -> pd.DataFrame:
    """Verarbeitet DataFrame für normalisierte Antworten."""
    print("Starte Datenvorverarbeitung...")
    data = input_data.copy()
    data['normalized_answer'] = np.nan
    norm_min, norm_max = 0.05, 0.95
    processed_categories = 0

    for category in data['Kategorie'].unique():
        cat_mask = data['Kategorie'] == category
        cat_data = data.loc[cat_mask, 'Antwort']
        numeric_answers = pd.to_numeric(cat_data, errors='coerce')
        num_numeric = numeric_answers.notna().sum()
        num_total = len(cat_data)

        if num_numeric > num_total * 0.8: # Numerische Kategorie
            print(f"  Kategorie '{category}': Verarbeite als numerisch.")
            min_val, max_val = numeric_answers.min(), numeric_answers.max()
            if pd.isna(min_val) or pd.isna(max_val):
                data.loc[cat_mask, 'normalized_answer'] = 0.5
            elif max_val == min_val:
                data.loc[cat_mask, 'normalized_answer'] = (norm_min + norm_max) / 2
            else:
                normalized = norm_min + (numeric_answers - min_val) * (norm_max - norm_min) / (max_val - min_val)
                data.loc[cat_mask, 'normalized_answer'] = normalized.fillna(0.5)
            processed_categories += 1
        else: # Text / Gemischt
            print(f"  Kategorie '{category}': Verarbeite als Text/Gemischt.")
            converted_values = []
            for answer in cat_data:
                text_value = convert_text_answer_to_numeric(answer)
                if text_value is not None:
                    converted_values.append(text_value)
                else:
                    try: # Prüfe auf isolierte Zahl
                        float(answer); converted_values.append(0.5) # Behandle als neutral
                    except (ValueError, TypeError):
                        converted_values.append(0.5) # Unbekannter Text -> neutral
            if len(converted_values) == len(data.loc[cat_mask]):
                data.loc[cat_mask, 'normalized_answer'] = converted_values
            else: data.loc[cat_mask, 'normalized_answer'] = 0.5 # Fallback
            processed_categories += 1

    data['normalized_answer'].fillna(0.5, inplace=True)
    print(f"Datenvorverarbeitung abgeschlossen. {processed_categories} Kategorien verarbeitet.")
    return data

# --- Netzwerk-Hilfsfunktionen (Rest) ---
# ... (social_influence, update_emotional_state, apply_contextual_factors, hebbian_learning unverändert) ...
def social_influence(nodes_list, social_network, influence_factor=0.05):
    for node in nodes_list:
        if not hasattr(node, 'label'): continue
        social_impact = social_network.get(node.label, 0) * influence_factor
        if hasattr(node, 'connections'):
            for conn in node.connections:
                 for source_node in nodes_list:
                     if hasattr(source_node, 'connections'):
                         for source_conn in source_node.connections:
                             if hasattr(source_conn, 'target_node') and source_conn.target_node == node:
                                 source_conn.weight += social_impact
                                 source_conn.weight = np.clip(source_conn.weight, 0, 1.0)

def update_emotional_state(emotional_state, base_activation_level=0.5, change_rate=0.02, volatility=0.05):
    change = (base_activation_level - emotional_state) * change_rate
    noise = np.random.normal(0, volatility)
    emotional_state += change + noise
    return np.clip(emotional_state, 0.7, 1.5)

def apply_contextual_factors(activation, node, context_factors):
    if not hasattr(node, 'label'): return activation
    context_factor = context_factors.get(node.label, 1.0) * random.uniform(0.95, 1.05)
    return np.clip(activation * context_factor, 0.0, 1.0)

def hebbian_learning(node, learning_rate=0.1, weight_limit=1.0, reg_factor=0.001):
    if getattr(node, 'activation', 0.0) < 0.1: return
    if hasattr(node, 'connections'):
        for connection in node.connections:
            target_activation = getattr(connection.target_node, 'activation', 0.0)
            if target_activation < 0.1: continue
            delta_weight = learning_rate * node.activation * target_activation
            connection.weight += delta_weight
            connection.weight -= reg_factor * connection.weight
            connection.weight = np.clip(connection.weight, 0.0, weight_limit)


# --- Klassen für Netzwerkstruktur ---
# ... (Connection, Node, MemoryNode unverändert) ...
class Connection:
    def __init__(self, target_node, weight=None):
        self.target_node = target_node
        self.weight = weight if weight is not None else random.uniform(0.05, 0.3)

class Node:
    def __init__(self, label: str, neuron_type: str = "excitatory"):
        self.label = label
        self.neuron_type = neuron_type
        self.connections = []
        self.activation = 0.0
        self.activation_sum = 0.0
        self.activation_count = 0
        self.activation_history = []
    def add_connection(self, target_node, weight=None):
        if not any(conn.target_node == target_node for conn in self.connections):
            self.connections.append(Connection(target_node, weight))

class MemoryNode(Node):
    def __init__(self, label: str, memory_type="short_term", neuron_type="excitatory"):
        super().__init__(label, neuron_type=neuron_type)
        self.memory_type = memory_type
        self.retention_times = {"short_term": 5, "mid_term": 20, "long_term": 100}
        self.retention_time = self.retention_times.get(memory_type, 20)
        self.time_in_memory = 0
        self.history_length_maps = {"short_term": 5, "mid_term": 20}
    def decay(self, decay_rate, context_factors={}, emotional_state=1.0):
        context_factor = context_factors.get(self.label, 1.0)
        emotional_factor = emotional_state
        decay_multipliers = {"short_term": 2.0, "mid_term": 1.0, "long_term": 0.5}
        decay_multiplier = decay_multipliers.get(self.memory_type, 1.0)
        factor = (1 - decay_rate * decay_multiplier * context_factor * emotional_factor)
        for conn in self.connections: conn.weight *= factor
    def promote(self, activation_threshold=DEFAULT_ACTIVATION_THRESHOLD_PROMOTION, history_length_map=None):
        if history_length_map is None: history_length_map = self.history_length_maps
        required_length = history_length_map.get(self.memory_type)
        if required_length is None: return
        if not hasattr(self, 'activation_history') or not self.activation_history: return
        if len(self.activation_history) >= required_length:
            avg_recent_activation = np.mean(self.activation_history[-required_length:])
            if avg_recent_activation > activation_threshold:
                if self.memory_type == "short_term":
                    self.memory_type = "mid_term"; self.retention_time = self.retention_times.get("mid_term", 20)
                    print(f"Info: Knoten '{self.label}' zu 'mid_term' befördert.")
                elif self.memory_type == "mid_term":
                    self.memory_type = "long_term"; self.retention_time = self.retention_times.get("long_term", 100)
                    print(f"Info: Knoten '{self.label}' zu 'long_term' befördert.")

# --- Spezialisierte Modul-Klassen ---
# ... (CortexCreativus, SimulatrixNeuralis, CortexCriticus, LimbusAffektus, MetaCognitio, CortexSocialis unverändert) ...
class CortexCreativus(Node):
    def __init__(self, label="Cortex Creativus", neuron_type=None):
        neuron_type = neuron_type if neuron_type else random_neuron_type()
        super().__init__(label, neuron_type=neuron_type)
    def generate_new_ideas(self, active_nodes):
        ideas = []
        sorted_nodes = sorted([n for n in active_nodes if getattr(n, 'activation', 0.0) > 0.6], key=lambda n: getattr(n, 'activation', 0.0), reverse=True)
        for i, node in enumerate(sorted_nodes[:3]): ideas.append(f"Idea_{i+1}_from_{node.label}")
        return ideas

class SimulatrixNeuralis(Node):
    def __init__(self, label="Simulatrix Neuralis", neuron_type=None):
        neuron_type = neuron_type if neuron_type else random_neuron_type()
        super().__init__(label, neuron_type=neuron_type)
    def simulate_scenarios(self, active_nodes):
        scenarios = []
        for node in active_nodes:
            if getattr(node, 'activation', 0.0) > 0.7: scenarios.append(f"Scenario_what_if_{node.label}_increases")
        return scenarios

class CortexCriticus(Node):
    def __init__(self, label="Cortex Criticus", neuron_type=None):
        neuron_type = neuron_type if neuron_type else random_neuron_type()
        super().__init__(label, neuron_type=neuron_type)
    def evaluate_ideas(self, ideas, current_network_state_nodes):
        evaluated = []
        activations = [getattr(n, 'activation', 0.0) for n in current_network_state_nodes]
        avg_activation = np.mean(activations) if activations else 0.5
        for idea in ideas:
            score = np.clip(random.uniform(0, 1) * avg_activation * 2, 0, 1)
            evaluated.append({"idea": idea, "score": score})
        return evaluated

class LimbusAffektus(Node):
    def __init__(self, label="Limbus Affektus", neuron_type=None):
        neuron_type = neuron_type if neuron_type else random_neuron_type()
        super().__init__(label, neuron_type=neuron_type)
    def update_emotion(self, current_emotional_state, active_nodes):
        positive_triggers = ["Höchstkurs"]; negative_triggers = ["Tiefstkurs"] # Beispiel
        pos_activation = np.mean([getattr(n,'activation',0.0) for n in active_nodes if hasattr(n, 'label') and n.label in positive_triggers and getattr(n,'activation',0.0) > 0.5]) or 0
        neg_activation = np.mean([getattr(n,'activation',0.0) for n in active_nodes if hasattr(n, 'label') and n.label in negative_triggers and getattr(n,'activation',0.0) > 0.5]) or 0
        change = (pos_activation - neg_activation) * 0.05
        new_emotional_state = current_emotional_state + change + np.random.normal(0, 0.02)
        return np.clip(new_emotional_state, 0.7, 1.5)

class MetaCognitio(Node):
    def __init__(self, label="Meta Cognitio", neuron_type=None):
        neuron_type = neuron_type if neuron_type else random_neuron_type()
        super().__init__(label, neuron_type=neuron_type)
    def optimize_learning_parameters(self, nodes_list, current_lr, current_dr):
        activations = [getattr(n, 'activation', 0.0) for n in nodes_list]
        avg_activation = np.mean(activations) if activations else 0.0
        new_lr, new_dr = current_lr, current_dr
        if avg_activation > 0.8: new_lr *= 0.98; new_dr *= 1.02
        elif avg_activation < 0.3: new_lr *= 1.01
        return np.clip(new_lr, 0.01, 1.0), np.clip(new_dr, 0.0005, 0.05)

class CortexSocialis(Node):
    def __init__(self, label="Cortex Socialis", neuron_type=None):
        neuron_type = neuron_type if neuron_type else random_neuron_type()
        super().__init__(label, neuron_type=neuron_type)
    def update_social_factors(self, social_network, active_nodes):
        if any(hasattr(n, 'label') and n.label == "Handelsvolumen" and getattr(n, 'activation', 0.0) > 0.7 for n in active_nodes): # Beispiel
            social_network["Handelsvolumen"] = min(social_network.get("Handelsvolumen", 0.5) * 1.1, 1.0)
        return social_network

# --- Verbindungsfunktion ---
def connect_new_brains_to_network(all_nodes, new_brains, module_category_weight=DEFAULT_MODULE_CATEGORY_WEIGHT):
    # ... (unverändert) ...
    category_nodes = [n for n in all_nodes if isinstance(n, MemoryNode)]
    for brain in new_brains:
        for node in all_nodes:
            if brain != node:
                brain.add_connection(node, weight=random.uniform(0.05, 0.2))
        for cat_node in category_nodes:
             brain.add_connection(cat_node, weight=module_category_weight)

# --- Visualisierungsfunktionen ---
# ... (plot_activation_and_weights, plot_dynamics, plot_new_brains_activation_comparison unverändert) ...
def plot_activation_and_weights(activation_history: dict, weights_history: dict) -> plt.Figure:
    print("Erstelle Plot-Daten: Aktivierungs- und Gewichtsentwicklung...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
    for label, activations in activation_history.items():
        if activations and (np.any(np.array(activations) != 0) or len(activations) < 5) :
            ax1.plot(range(1, len(activations) + 1), activations, label=label, alpha=0.8)
    ax1.set_title("Entwicklung der Aktivierungen"); ax1.set_xlabel("Epoche"); ax1.set_ylabel("Aktivierung")
    ax1.legend(fontsize='small', loc='best'); ax1.grid(True)
    max_weights_to_plot = 30; plotted_weights = 0
    if weights_history:
        sorted_weights_keys = sorted(weights_history.keys())
        for label in sorted_weights_keys:
            weights = weights_history.get(label, [])
            if weights and plotted_weights < max_weights_to_plot:
                ax2.plot(range(1, len(weights) + 1), weights, label=label, alpha=0.6); plotted_weights += 1
            if plotted_weights >= max_weights_to_plot: break
    ax2.set_title(f"Entwicklung der Gewichte (max. {max_weights_to_plot})"); ax2.set_xlabel("Epoche"); ax2.set_ylabel("Gewicht")
    if plotted_weights > 0: ax2.legend(fontsize='small', bbox_to_anchor=(1.04, 1), loc='upper left')
    ax2.grid(True); plt.tight_layout(rect=[0, 0, 0.85, 1])
    return fig

def plot_dynamics(activation_history, weights_history) -> plt.Figure:
    print("Erstelle Plot-Daten: Netzwerk-Dynamiken...")
    fig, axs = plt.subplots(2, 2, figsize=(16, 12)); axs = axs.flatten()
    plotted_activation = False
    for label, activations in activation_history.items():
        if activations and (np.any(np.array(activations) != 0) or len(activations) < 5):
             axs[0].plot(range(1, len(activations) + 1), activations, label=label, alpha=0.8); plotted_activation = True
    axs[0].set_title("Entwicklung der Aktivierungen"); axs[0].set_xlabel("Epoche"); axs[0].set_ylabel("Aktivierung")
    if plotted_activation: axs[0].legend(fontsize='x-small')
    axs[0].grid(True)
    max_weights_to_plot = 20; plotted_weights = 0
    if weights_history:
        sorted_weights_keys = sorted(weights_history.keys())
        for label in sorted_weights_keys:
            weights = weights_history.get(label, [])
            if weights and plotted_weights < max_weights_to_plot:
                axs[1].plot(range(1, len(weights) + 1), weights, label=label, alpha=0.6); plotted_weights += 1
            elif plotted_weights >= max_weights_to_plot: break
    axs[1].set_title(f"Entwicklung der Gewichte (max. {max_weights_to_plot})"); axs[1].set_xlabel("Epoche"); axs[1].set_ylabel("Gewicht")
    if plotted_weights > 0: axs[1].legend(bbox_to_anchor=(1.04, 1), loc='upper left', fontsize='x-small')
    axs[1].grid(True)
    avg_weights_per_epoch = []
    num_epochs = max((len(w) for w in weights_history.values()), default=0) if weights_history else 0
    for epoch_idx in range(num_epochs):
        epoch_weights = [weights[epoch_idx] for weights in weights_history.values() if len(weights) > epoch_idx]
        avg_weights_per_epoch.append(np.mean(epoch_weights) if epoch_weights else np.nan)
    if num_epochs > 0 : axs[2].plot(range(1, num_epochs + 1), avg_weights_per_epoch, label="Durchschnitt aller Gewichte")
    axs[2].set_title("Durchschnittliche Gewichtsentwicklung"); axs[2].set_xlabel("Epoche"); axs[2].set_ylabel("Durchschnittliches Gewicht"); axs[2].grid(True)
    std_activations_per_epoch = []
    num_epochs_act = max((len(a) for a in activation_history.values()), default=0) if activation_history else 0
    for epoch_idx in range(num_epochs_act):
        epoch_activations = [history[epoch_idx] for history in activation_history.values() if len(history) > epoch_idx]
        std_dev = np.std(epoch_activations) if len(epoch_activations) > 1 else 0.0
        std_activations_per_epoch.append(std_dev)
    if num_epochs_act > 0: axs[3].plot(range(1, num_epochs_act + 1), std_activations_per_epoch, label="StdAbw über Knoten")
    axs[3].set_title("Stabilität der Aktivierungen"); axs[3].set_xlabel("Epoche"); axs[3].set_ylabel("Standardabweichung"); axs[3].grid(True)
    plt.tight_layout(rect=[0, 0, 0.9, 1])
    return fig

def plot_new_brains_activation_comparison(new_brains_activation_history) -> plt.Figure:
    print("Erstelle Plot-Daten: Modul-Aktivierungsvergleich...")
    fig = plt.figure(figsize=(12, 8))
    plotted = False
    for label, activations in new_brains_activation_history.items():
        if activations and (np.any(np.array(activations) != 0) or len(activations) < 5):
             plt.plot(range(1, len(activations) + 1), activations, label=label); plotted = True
    plt.title("Vergleich der Aktivierungen der neuen Gehirnmodule"); plt.xlabel("Epoche"); plt.ylabel("Aktivierung")
    if plotted: plt.legend()
    plt.grid(True)
    return fig

# --- Netzwerk-Initialisierung ---
def initialize_quiz_network(categories):
    # ... (unverändert) ...
    category_nodes = [MemoryNode(c, memory_type="short_term", neuron_type="excitatory") for c in categories]
    for node in category_nodes: node.activation_history = []
    return category_nodes

# --- Signalpropagation ---
def propagate_signal(node, input_signal, emotion_weights, emotional_state=1.0, context_factors=None):
    # ... (unverändert, mit NaN-Check) ...
    try:
        node.activation = float(input_signal)
        if np.isnan(node.activation): node.activation = 0.0
    except (ValueError, TypeError): node.activation = 0.0
    node.activation = add_activation_noise(node.activation)
    node.activation = np.clip(node.activation, 0.0, 1.0)
    if hasattr(node, 'connections'):
        for connection in node.connections:
            signal_strength = node.activation * connection.weight
            target_node = connection.target_node
            if hasattr(target_node, 'activation_sum'):
                if node.neuron_type == "inhibitory": target_node.activation_sum -= signal_strength
                else: target_node.activation_sum += signal_strength
                target_node.activation_sum = np.clip(target_node.activation_sum, -10, 10)

# --- Simulation ---
def simulate_learning(input_data, category_nodes, new_brains,
                      epochs=DEFAULT_EPOCHS, learning_rate=DEFAULT_LEARNING_RATE,
                      reward_interval=DEFAULT_REWARD_INTERVAL, decay_rate=DEFAULT_DECAY_RATE,
                      initial_emotional_state=1.0, context_factors=None):
    # ... (weitgehend unverändert, ruft preprocess_data auf) ...
    data = preprocess_data(input_data)
    if 'normalized_answer' not in data.columns: return {}, {}, []
    if context_factors is None: context_factors = {}
    core_network_nodes = category_nodes + new_brains
    module_dict = {brain.label: brain for brain in new_brains}
    weights_history = {}
    valid_core_labels = {node.label for node in core_network_nodes if hasattr(node, 'label')}
    activation_history = {label: [] for label in valid_core_labels}
    module_outputs_log = {label: [] for label in module_dict.keys()}
    interpretation_log = []

    for node in core_network_nodes:
        if not hasattr(node, 'activation_history'): node.activation_history = []
        if not hasattr(node, 'activation_sum'): node.activation_sum = 0.0
        if not hasattr(node, 'neuron_type'): node.neuron_type = random_neuron_type()

    question_nodes = []
    for idx, row in data.iterrows():
        q_label = f"Q_{idx}_{row['Frage'][:20]}"
        q_node = Node(q_label, neuron_type=random_neuron_type())
        question_nodes.append(q_node)
        category_label = row['Kategorie'].strip()
        category_node = next((c for c in category_nodes if hasattr(c, 'label') and c.label == category_label), None)
        if category_node: q_node.add_connection(category_node, weight=0.7)
        else: print(f"Warnung: Kategorie '{category_label}' nicht gefunden für Frage '{row['Frage']}'.")

    all_nodes_sim = core_network_nodes + question_nodes
    global question_nodes_global; question_nodes_global = question_nodes

    for node in all_nodes_sim: # Weights History Initialisierung
        if hasattr(node, 'connections'):
            for conn in node.connections:
                 source_label = getattr(node, 'label', None); target_label = getattr(conn.target_node, 'label', None)
                 if source_label and target_label: weights_history[f"{source_label} → {target_label}"] = []

    emotion_weights = {node.label: 1.0 for node in all_nodes_sim if hasattr(node, 'label')}
    social_network = {node.label: random.uniform(0.1, 0.5) for node in category_nodes}
    current_emotional_state = initial_emotional_state; current_lr = learning_rate; current_dr = decay_rate

    for epoch in tqdm(range(epochs), desc="Simulating Learning"): # Simulationsschleife
        for node in core_network_nodes: node.activation_sum = 0.0 # Reset Sums

        for idx, q_node in enumerate(question_nodes): # Input Phase
            normalized_answer = data.iloc[idx]['normalized_answer']
            propagate_signal(q_node, normalized_answer, emotion_weights, current_emotional_state, context_factors)

        active_category_nodes_this_epoch = [] # Core Network Update
        for node in core_network_nodes:
            node.activation = sigmoid(node.activation_sum)
            node.activation = apply_emotion_weight(node.activation, getattr(node, 'label', None), emotion_weights, current_emotional_state)
            if context_factors: node.activation = apply_contextual_factors(node.activation, node, context_factors)
            node.activation = add_activation_noise(node.activation)
            node.activation = np.clip(node.activation, 0.0, 1.0)
            if hasattr(node, 'label') and node.label in activation_history: activation_history[node.label].append(node.activation)
            if node in category_nodes and node.activation > 0.5: active_category_nodes_this_epoch.append(node)
            if isinstance(node, MemoryNode):
                 if hasattr(node, 'activation_history'): node.activation_history.append(node.activation)
                 node.promote()

        # Modul-Operationen... (unverändert)
        if module_dict.get("Cortex Creativus"): ideas = module_dict["Cortex Creativus"].generate_new_ideas(active_category_nodes_this_epoch); module_outputs_log["Cortex Creativus"].append(ideas)
        if module_dict.get("Simulatrix Neuralis"): scenarios = module_dict["Simulatrix Neuralis"].simulate_scenarios(active_category_nodes_this_epoch); module_outputs_log["Simulatrix Neuralis"].append(scenarios)
        if module_dict.get("Cortex Criticus") and module_outputs_log.get("Cortex Creativus") and module_outputs_log["Cortex Creativus"][-1]: evaluated = module_dict["Cortex Criticus"].evaluate_ideas(module_outputs_log["Cortex Creativus"][-1], category_nodes); module_outputs_log["Cortex Criticus"].append(evaluated)
        if module_dict.get("Limbus Affektus"): current_emotional_state = module_dict["Limbus Affektus"].update_emotion(current_emotional_state, category_nodes); module_outputs_log["Limbus Affektus"].append(f"State: {current_emotional_state:.3f}")
        if module_dict.get("Meta Cognitio"): current_lr, current_dr = module_dict["Meta Cognitio"].optimize_learning_parameters(core_network_nodes, current_lr, current_dr); module_outputs_log["Meta Cognitio"].append(f"LR:{current_lr:.4f},DR:{current_dr:.5f}")
        if module_dict.get("Cortex Socialis"): social_network = module_dict["Cortex Socialis"].update_social_factors(social_network, category_nodes); module_outputs_log["Cortex Socialis"].append("Factors updated"); social_influence(category_nodes, social_network)

        # Lernen & Adaptation... (unverändert)
        for node in all_nodes_sim: hebbian_learning(node, current_lr, reg_factor=0.001)
        if (epoch + 1) % reward_interval == 0 and category_nodes:
             target_category_label = random.choice([n.label for n in category_nodes if hasattr(n,'label')])
             reward_connections(all_nodes_sim, target_category_label, reward_factor=0.05)
        decay_weights(all_nodes_sim, current_dr)

        # Gewichtshistorie aufzeichnen... (unverändert)
        for node in all_nodes_sim:
            if hasattr(node, 'connections'):
                for conn in node.connections:
                    source_label = getattr(node, 'label', None); target_label = getattr(conn.target_node, 'label', None)
                    if source_label and target_label:
                         history_key = f"{source_label} → {target_label}"
                         if history_key in weights_history: weights_history[history_key].append(conn.weight)

        interpretation_log.append(interpret_epoch_state(epoch, category_nodes, new_brains, module_outputs_log, activation_history)) # Interpretation

    return activation_history, weights_history, interpretation_log


# --- Interpretation & Report ---
def interpret_epoch_state(epoch, category_nodes, modules, module_outputs, activation_history):
    # ... (unverändert) ...
    interpretation = {'epoch': epoch + 1}
    sorted_cats = sorted(category_nodes, key=lambda n: getattr(n, 'activation', 0.0), reverse=True)
    if sorted_cats:
        interpretation['dominant_category'] = getattr(sorted_cats[0], 'label', 'Unbekannt')
        interpretation['dominant_activation'] = getattr(sorted_cats[0], 'activation', 0.0)
        interpretation['category_ranking'] = [(getattr(n, 'label', 'Unbekannt'), getattr(n, 'activation', 0.0)) for n in sorted_cats]
    else: interpretation['dominant_category'] = 'N/A'; interpretation['dominant_activation'] = 0.0; interpretation['category_ranking'] = []
    interpretation['module_activations'] = {getattr(m, 'label', 'Unbekannt'): getattr(m, 'activation', 0.0) for m in modules}
    return interpretation

def generate_final_report(category_nodes, modules, data, interpretation_log):
    # ... (unverändert, gibt jetzt String zurück) ...
    print("\n--- NeuroPersona Analysebericht ---")
    report_lines = ["**NeuroPersona Analysebericht**\n"]
    threshold_high = 0.65; threshold_low = 0.35
    report_lines.append("**Finale Netzwerk-Tendenzen (Kategorien):**\n")
    sorted_categories = sorted(category_nodes, key=lambda n: getattr(n, 'activation', 0.0), reverse=True)
    for node in sorted_categories:
        category = node.label; activation = getattr(node, 'activation', 0.0)
        if activation >= threshold_high: tendency_label = "Stark Aktiviert / Hohe Relevanz"
        elif activation <= threshold_low: tendency_label = "Schwach Aktiviert / Geringe Relevanz"
        else: tendency_label = "Mittel Aktiviert / Neutrale Relevanz"
        line = f"- **{category}:** {tendency_label} (Aktivierung: {activation:.3f})."
        try: q_example = data[data['Kategorie'] == category]['Frage'].iloc[0]; line += f" (Bsp: '{q_example}')"
        except IndexError: line += " (Keine Bsp.-Frage)"
        report_lines.append(line)
    if interpretation_log:
        report_lines.append("\n**Verlaufseindrücke:**")
        dominant_cats = [log.get('dominant_category') for log in interpretation_log if log.get('dominant_category') != 'N/A']
        if dominant_cats:
             try: most_frequent = Counter(dominant_cats).most_common(1)[0][0]; report_lines.append(f"- '{most_frequent}' war häufig dominant.")
             except IndexError: report_lines.append("- Keine dominante Kategorie im Verlauf festgestellt.")
        else: report_lines.append("- Keine dominante Kategorie im Verlauf festgestellt.")
    report_lines.append("\n**Finaler Zustand der kognitiven Module:**")
    for module in modules:
        activation = getattr(module, 'activation', 0.0); neuron_type = getattr(module, 'neuron_type', '?')
        if np.isnan(activation): activation = 0.0
        report_lines.append(f"- {module.label} (Typ: {neuron_type}): {activation:.3f}")
    if sorted_categories:
        top_cat_node = sorted_categories[0]; top_cat = getattr(top_cat_node, 'label', 'Unbekannt'); top_act = getattr(top_cat_node, 'activation', 0.0)
        if top_act > threshold_high: report_lines.append(f"\n**Gesamteinschätzung:** Klare Tendenz bzgl. **{top_cat}**.")
        elif top_act < threshold_low and len(sorted_categories) > 1: second_cat_node = sorted_categories[1]; second_cat = getattr(second_cat_node, 'label', 'Unbekannt'); report_lines.append(f"\n**Gesamteinschätzung:** Geringe Aktivierung insgesamt. '{top_cat}' und '{second_cat}' relativ am höchsten.")
        elif top_cat != 'Unbekannt': report_lines.append(f"\n**Gesamteinschätzung:** Keine starke Dominanz. {top_cat} am aktivsten.")
        else: report_lines.append(f"\n**Gesamteinschätzung:** Keine klare Tendenz oder dominante Kategorie.")
    else: report_lines.append(f"\n**Gesamteinschätzung:** Keine Kategorien gefunden oder analysiert.")
    final_report = "\n".join(report_lines); print(final_report); return final_report

def filter_module_history(activation_history, module_labels):
    # ... (unverändert) ...
    return {label: activation_history[label] for label in module_labels if label in activation_history}

# --- Modell Speichern/Laden ---
# ... (save_model, load_model leicht angepasst für Robustheit, aber Kern unverändert) ...
def save_model(nodes_list, filename=MODEL_FILENAME):
    model_data = {"nodes": [], "connections": []}
    valid_nodes = [node for node in nodes_list if hasattr(node, 'label')]
    node_labels = {node.label for node in valid_nodes}
    print(f"Speichere {len(valid_nodes)} Knoten in {filename}...")
    for node in valid_nodes:
        node_info = {"label": node.label, "activation": getattr(node, 'activation', 0.0), "neuron_type": getattr(node, 'neuron_type', "excitatory")}
        if isinstance(node, MemoryNode): node_info["memory_type"] = node.memory_type; node_info["time_in_memory"] = node.time_in_memory
        model_data["nodes"].append(node_info)
        if hasattr(node, 'connections'):
            for conn in node.connections:
                target_label = getattr(conn.target_node, 'label', None)
                if target_label and target_label in node_labels: model_data["connections"].append({"source": node.label, "target": target_label, "weight": conn.weight})
    try:
        with open(filename, "w", encoding='utf-8') as file: json.dump(model_data, file, indent=4, ensure_ascii=False)
        print(f"Modell erfolgreich gespeichert in {filename}")
    except (IOError, TypeError) as e: print(f"FEHLER beim Speichern des Modells: {e}"); messagebox.showerror("Speicherfehler", f"Modell konnte nicht gespeichert werden:\n{e}")

def load_model(filename=MODEL_FILENAME, categories=None):
    if not os.path.exists(filename): print(f"Modelldatei {filename} nicht gefunden."); return None
    print(f"Lade Modell aus {filename}...")
    try:
        with open(filename, "r", encoding='utf-8') as file: model_data = json.load(file)
    except (IOError, json.JSONDecodeError) as e: print(f"FEHLER beim Laden/Parsen: {e}"); return None
    loaded_nodes, node_dict = [], {}; module_classes = {"Cortex Creativus": CortexCreativus, "Simulatrix Neuralis": SimulatrixNeuralis, "Cortex Criticus": CortexCriticus, "Limbus Affektus": LimbusAffektus, "Meta Cognitio": MetaCognitio, "Cortex Socialis": CortexSocialis}
    if categories is not None and hasattr(categories, '__iter__') and not isinstance(categories, str):
         try: categories_set = set(categories)
         except TypeError: categories_set = set(); print("Warnung: 'categories' konnte nicht in ein Set konvertiert werden.")
    else: categories_set = set(); print("Warnung: 'categories' ist kein Set oder iterierbar.")
    for node_data in model_data.get("nodes", []):
        label = node_data.get("label");
        if not label: print("Warnung: Überspringe Knoten ohne Label beim Laden."); continue
        neuron_type = node_data.get("neuron_type", "excitatory")
        if label in categories_set: node_class = MemoryNode
        elif label in module_classes: node_class = module_classes[label]
        else: node_class = Node
        node = node_class(label, neuron_type=neuron_type); node.activation = node_data.get("activation", 0.0)
        if not hasattr(node, 'activation_history'): node.activation_history = []
        if isinstance(node, MemoryNode):
            node.memory_type = node_data.get("memory_type", "short_term"); node.time_in_memory = node_data.get("time_in_memory", 0)
            node.retention_time = node.retention_times.get(node.memory_type, node.retention_times["mid_term"])
        loaded_nodes.append(node); node_dict[label] = node
    connection_count, missing_nodes = 0, set(); connections_data = model_data.get("connections", [])
    print(f"Versuche {len(connections_data)} Verbindungen wiederherzustellen...")
    for conn_data in connections_data:
        source_label, target_label, weight = conn_data.get("source"), conn_data.get("target"), conn_data.get("weight")
        source_node, target_node = node_dict.get(source_label), node_dict.get(target_label)
        if source_node and target_node and weight is not None:
            if not any(conn.target_node == target_node for conn in getattr(source_node, 'connections', [])):
                 source_node.add_connection(target_node, weight); connection_count += 1
        else:
            if not source_node: missing_nodes.add(source_label)
            if not target_node: missing_nodes.add(target_label)
    if missing_nodes: print(f"Warnung: Verbindungen für fehlende Knoten nicht wiederhergestellt: {missing_nodes}")
    print(f"{len(loaded_nodes)} Knoten und {connection_count} Verbindungen geladen/wiederhergestellt.")
    return loaded_nodes

# --- Hauptfunktionen ---
data = None
question_nodes_global = []

def main():
    # ... (unverändert, lädt Daten, startet GUI) ...
    start_time = time.time()
    print("Starte NeuroPersona...")
    global data
    csv_file = "data.csv"
    try:
        data = pd.read_csv(csv_file); print(f"CSV-Datei '{csv_file}' geladen ({len(data)} Zeilen).")
        required_columns = ['Frage', 'Antwort', 'Kategorie']
        if not all(col in data.columns for col in required_columns):
             missing_cols = [col for col in required_columns if col not in data.columns]; messagebox.showerror("CSV Fehler", f"CSV fehlt Spalten: {missing_cols}"); print(f"FEHLER: CSV fehlt Spalten: {missing_cols}"); return
        initial_rows = len(data); data.dropna(subset=required_columns, inplace=True)
        if len(data) < initial_rows: print(f"Warnung: {initial_rows - len(data)} Zeilen mit fehlenden Werten entfernt.")
    except FileNotFoundError: print(f"FEHLER: CSV '{csv_file}' nicht gefunden."); root = tk.Tk(); root.withdraw(); messagebox.showerror("Fehler", f"CSV '{csv_file}' nicht gefunden."); root.destroy(); return
    except Exception as e: print(f"FEHLER beim Lesen der CSV: {e}"); root = tk.Tk(); root.withdraw(); messagebox.showerror("Fehler", f"Fehler Lesen CSV: {e}"); root.destroy(); return
    if data is None or data.empty: print("Keine gültigen Daten."); messagebox.showerror("Fehler", "Keine gültigen Daten in CSV."); return
    start_gui(data)

def safe_show_plot(fig=None):
    # ... (unverändert) ...
    if threading.current_thread() is not threading.main_thread(): print("Warnung: safe_show_plot nicht im Main Thread."); return
    try:
        if fig and fig.axes: plt.pause(0.1);
        elif fig: plt.close(fig)
        elif plt.get_fignums(): plt.show(block=False); plt.pause(0.1);
    except Exception as e: print(f"Fehler beim Anzeigen des Plots: {e}")

def get_important_categories(category_nodes):
    # ... (unverändert) ...
    important_categories = []
    for node in category_nodes:
        act = getattr(node, 'activation', 0.0)
        if act >= 0.8: importance = "sehr hoch"
        elif act >= 0.6: importance = "hoch"
        elif act >= 0.4: importance = "mittel"
        elif act >= 0.2: importance = "gering"
        else: importance = "sehr gering"
        important_categories.append((node.label, importance))
    important_categories.sort(key=lambda item: getattr(next((n for n in category_nodes if n.label == item[0]), None), 'activation', 0.0), reverse=True)
    return important_categories


def run_simulation_from_gui(learning_rate, decay_rate, reward_interval, epochs, root, module_category_weights):
    """Führt die Simulation aus und gibt Ergebnisse ZURÜCK (inkl. final_report_text)."""
    # ... (Netzwerkaufbau/Laden unverändert) ...
    start_time = time.time()
    print(f"\n--- Starte Simulation: LR={learning_rate}, DR={decay_rate}, RI={reward_interval}, Ep={epochs} ---")
    global data
    if data is None or data.empty : messagebox.showerror("Fehler", "Simulationsdaten nicht verfügbar."); return None, None, None, None
    categories = data['Kategorie'].unique()
    loaded_nodes = load_model(MODEL_FILENAME, categories=categories)
    category_nodes, new_brains = [], []
    module_labels_expected = {"Cortex Creativus", "Simulatrix Neuralis", "Cortex Criticus", "Limbus Affektus", "Meta Cognitio", "Cortex Socialis"}
    if loaded_nodes:
        print("Prüfe geladenes Modell..."); category_nodes = [n for n in loaded_nodes if isinstance(n, MemoryNode) and hasattr(n, 'label') and n.label in categories]; new_brains = [n for n in loaded_nodes if n not in category_nodes and hasattr(n, 'label') and n.label in module_labels_expected]
        other_loaded_nodes = [n for n in loaded_nodes if n not in category_nodes and n not in new_brains];
        if other_loaded_nodes: print(f"Warnung: {len(other_loaded_nodes)} unklassifizierte Knoten geladen.")
        loaded_module_labels = {n.label for n in new_brains}; print(f"Modell geladen: {len(category_nodes)} Kat, {len(new_brains)} Module.")
        valid_load = True
        if loaded_module_labels != module_labels_expected: print(f"Warnung: Modul-Mismatch! Re-Initialisiere."); valid_load = False
        if set(n.label for n in category_nodes) != set(categories): print(f"Warnung: Kategorie-Mismatch! Re-Initialisiere."); valid_load = False
        if not valid_load: loaded_nodes = None
    if loaded_nodes is None:
        print("Initialisiere neues Netzwerk..."); category_nodes = initialize_quiz_network(categories); new_brains = [CortexCreativus(), SimulatrixNeuralis(), CortexCriticus(), LimbusAffektus(), MetaCognitio(), CortexSocialis()]; print(f"Neues Netzwerk: {len(category_nodes)} Kat, {len(new_brains)} Module.")
    if not category_nodes: messagebox.showerror("Fehler", "Kategorie-Initialisierung fehlgeschlagen."); return None, None, None, None
    module_labels = [m.label for m in new_brains]
    all_nodes_connect = category_nodes + new_brains
    connect_new_brains_to_network_with_gui_weights(all_nodes_connect, new_brains, module_category_weights); print("Netzwerk verbunden (mit GUI-Gewichten).")

    # --- Simulation ---
    activation_history, weights_history, interpretation_log = simulate_learning(data, category_nodes, new_brains, int(epochs), float(learning_rate), int(reward_interval), float(decay_rate))

    # --- Nachbereitung ---
    final_nodes_to_save = category_nodes + new_brains; save_model(final_nodes_to_save, MODEL_FILENAME)
    final_report_text = generate_final_report(category_nodes, new_brains, data, interpretation_log) # ### Bericht wird hier generiert

    # Regelbasierte Empfehlung (Beispiel)
    sorted_categories = sorted(category_nodes, key=lambda n: getattr(n, 'activation', 0.0), reverse=True)
    final_recommendation = "Unbekannt"
    if sorted_categories:
         top_category_node = sorted_categories[0]; top_category = getattr(top_category_node, 'label', 'Unbekannt'); top_activation = getattr(top_category_node, 'activation', 0.0)
         if top_category == "Höchstkurs" and top_activation > 0.7: final_recommendation = "Empfehlung"
         elif top_category == "Tiefstkurs" and top_activation > 0.7: final_recommendation = "Abraten"
         elif top_activation < 0.4: final_recommendation = "Abwarten"
         else: final_recommendation = "Abwarten"

    important_categories = get_important_categories(category_nodes)
    create_html_report(final_report_text, final_recommendation, interpretation_log, important_categories)
    end_time = time.time(); print(f"GUI Simulation abgeschlossen ({end_time - start_time:.2f}s)")

    # ### NEU: Gib auch den generierten Textbericht zurück ###
    return activation_history, weights_history, module_labels, final_report_text


def connect_new_brains_to_network_with_gui_weights(all_nodes, new_brains, module_category_weights):
    # ... (unverändert) ...
    category_nodes = [n for n in all_nodes if isinstance(n, MemoryNode) and hasattr(n, 'label')]; module_nodes = [n for n in new_brains if hasattr(n, 'label')]
    print(f"Verbinde {len(module_nodes)} Module mit {len(category_nodes)} Kategorien...");
    for brain in module_nodes:
        for other_brain in module_nodes:
            if brain != other_brain: brain.add_connection(other_brain, weight=random.uniform(0.01, 0.1))
        for cat_node in category_nodes:
            key = (brain.label, cat_node.label); weight = module_category_weights.get(key, DEFAULT_MODULE_CATEGORY_WEIGHT); brain.add_connection(cat_node, weight=weight)
            incoming_weight = random.uniform(0.05, 0.2); cat_node.add_connection(brain, weight=incoming_weight)

# --- Plot-Generierung ---
def generate_and_display_plots(plot_data, root, status_label):
    # ... (unverändert, aber bekommt plot_data-Tupel) ...
    if not plot_data or len(plot_data) < 3: # Prüfe ob Tupel valide ist
        print("Ungültige Plot-Daten erhalten."); status_label.config(text="Status: Fehler bei Plotdaten."); return
    activation_history, weights_history, module_labels = plot_data[:3] # Nimm nur die ersten 3 Elemente
    if not activation_history and not weights_history: print("Leere Historien."); status_label.config(text="Status: Leere Ergebnisse."); return
    print("Erstelle, speichere, zeige Plots..."); status_label.config(text="Status: Erstelle Plots...")
    module_activation_history = filter_module_history(activation_history, module_labels); plots_dir = "plots"; os.makedirs(plots_dir, exist_ok=True)
    fig_act_weights, fig_dynamics, fig_comparison = None, None, None
    try:
        if any(act for act in activation_history.values()) or any(w for w in weights_history.values()):
             fig_act_weights = plot_activation_and_weights(activation_history, weights_history)
             if fig_act_weights: fig_act_weights.savefig(os.path.join(plots_dir, "aktivierung_und_gewichte.png")); print("Plot 1 gespeichert.")
    except Exception as e: print(f"Fehler Plot 1: {e}")
    try:
        if any(act for act in activation_history.values()) or any(w for w in weights_history.values()):
            fig_dynamics = plot_dynamics(activation_history, weights_history)
            if fig_dynamics: fig_dynamics.savefig(os.path.join(plots_dir, "netzwerk_dynamiken.png")); print("Plot 2 gespeichert.")
    except Exception as e: print(f"Fehler Plot 2: {e}")
    if module_activation_history and any(act for act in module_activation_history.values()):
        try:
            fig_comparison = plot_new_brains_activation_comparison(module_activation_history)
            if fig_comparison: fig_comparison.savefig(os.path.join(plots_dir, "modul_vergleich.png")); print("Plot 3 gespeichert.")
        except Exception as e: print(f"Fehler Plot 3: {e}")
    print("Zeige Plots...");
    if fig_act_weights: safe_show_plot(fig_act_weights)
    if fig_dynamics: safe_show_plot(fig_dynamics)
    if fig_comparison: safe_show_plot(fig_comparison)
    print("Visualisierungen abgeschlossen.")
    # Status-Update und Info-Box kommen NACH der Gemini-Analyse

# --- ### NEU: Gemini API Interaktion ### ---
def get_gemini_report(user_prompt: str, neuro_persona_report: str) -> str:
    """
    Ruft die Gemini API auf, um einen Bericht basierend auf dem User-Prompt
    und dem NeuroPersona-Analyseergebnis zu generieren.
    """
    if not gemini_available:
        return "Fehler: Google Generative AI SDK nicht installiert."

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        error_msg = ("Fehler: Gemini API Key nicht gefunden. "
                     "Bitte setzen Sie die Umgebungsvariable 'GEMINI_API_KEY'.")
        print(error_msg)
        # Optional: Zeige Fehler auch in GUI an
        # messagebox.showerror("Gemini Fehler", error_msg)
        return error_msg

    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-2.0-flash-exp') # Oder ein spezifischeres Modell wählen

        # Kombiniere User-Prompt und NeuroPersona-Bericht zu einem klaren Prompt für Gemini
        combined_prompt = f"""
        Analysiere die folgenden Ergebnisse der NeuroPersona-Simulation im Kontext der Benutzeranfrage.
        Erstelle einen detaillierten, gut strukturierten Bericht in deutscher Sprache als Markdown.

        **Benutzeranfrage / Analysefokus:**
        {user_prompt}

        **Ergebnisse der NeuroPersona Simulation:**
        ```markdown
        {neuro_persona_report}
        ```

        **Deine Aufgabe:**
        Generiere einen umfassenden Bericht, der:
        1.  Die Benutzeranfrage aufgreift.
        2.  Die wichtigsten Ergebnisse der NeuroPersona-Simulation (dominante Kategorien, Modulzustände, Verlauf) zusammenfasst.
        3.  Diese Ergebnisse im Licht der Benutzeranfrage interpretiert.
        4.  Mögliche Implikationen, Einschränkungen oder nächste Schritte aufzeigt, basierend auf der Simulation UND der Benutzeranfrage.
        5.  Den emergenten, bio-inspirierten Charakter der NeuroPersona-Simulation berücksichtigt, falls relevant für die Interpretation.
        """

        print("\n--- Sende Anfrage an Gemini API ---")
        # Sicherheitseinstellungen (optional, anpassen bei Content-Filter-Problemen)
        # safety_settings = [
        #     {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
        #     {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
        #     {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
        #     {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
        # ]
        # response = model.generate_content(combined_prompt, safety_settings=safety_settings)

        response = model.generate_content(combined_prompt)

        print("--- Antwort von Gemini API erhalten ---")

        # Extrahiere und bereinige den Text
        gemini_text = response.text.strip()

        # Speichere den Gemini-Bericht optional in einer Datei
        try:
            with open(GEMINI_REPORT_FILENAME, "w", encoding="utf-8") as f:
                f.write(f"# Gemini Analyse Bericht\n\n")
                f.write(f"**Benutzeranfrage:**\n{user_prompt}\n\n")
                f.write(f"**NeuroPersona Basisbericht:**\n```markdown\n{neuro_persona_report}\n```\n\n")
                f.write(f"---\n\n")
                f.write(f"# Von Gemini generierter Bericht:\n\n")
                f.write(gemini_text)
            print(f"Gemini-Bericht gespeichert in: {GEMINI_REPORT_FILENAME}")
        except IOError as e:
            print(f"Fehler beim Speichern des Gemini-Berichts: {e}")

        return gemini_text

    except Exception as e:
        error_msg = f"Fehler bei der Kommunikation mit der Gemini API: {e}"
        print(error_msg)
        # Optional: Zeige Fehler auch in GUI an
        # messagebox.showerror("Gemini API Fehler", error_msg)
        return error_msg

# --- ### NEU: Funktion zum Anzeigen des Gemini-Berichts ### ---
def display_gemini_report(report_text: str, parent_root: tk.Tk):
    """Zeigt den Gemini-Bericht in einem neuen Fenster an."""
    if not report_text:
        messagebox.showwarning("Gemini Analyse", "Kein Bericht von Gemini erhalten.")
        return

    report_window = tk.Toplevel(parent_root)
    report_window.title("Gemini Analyse Bericht")
    report_window.geometry("800x600")

    # ScrolledText Widget für den Bericht
    st_widget = scrolledtext.ScrolledText(report_window, wrap=tk.WORD, padx=10, pady=10)
    st_widget.pack(fill=tk.BOTH, expand=True)

    # Füge den Text ein
    st_widget.insert(tk.END, report_text)

    # Mache das Textfeld schreibgeschützt
    st_widget.configure(state='disabled')

    # Optional: Button zum Schließen
    close_button = ttk.Button(report_window, text="Schließen", command=report_window.destroy)
    close_button.pack(pady=10)

    # Fenster in den Vordergrund bringen
    report_window.lift()
    report_window.focus_force()


# --- GUI Startfunktion (angepasst mit Gemini-Prompt-Feld) ---
weight_sliders = {}
entry_widgets = {}
gemini_prompt_text_widget = None # ### NEU: Globale Referenz für Prompt-Widget

def start_gui(loaded_data):
    global data, weight_sliders, entry_widgets, gemini_prompt_text_widget # ### NEU ###
    data = loaded_data
    if data is None or data.empty: messagebox.showerror("Fehler", "Keine Daten."); return

    root = tk.Tk(); root.title("NeuroPersona Simulation GUI"); root.geometry("900x750") # Höhe leicht erhöht
    style = ttk.Style(); style.theme_use('clam') # Oder anderes Theme

    main_frame = ttk.Frame(root); main_frame.pack(fill=tk.BOTH, expand=True)
    canvas = tk.Canvas(main_frame); canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    scrollbar = ttk.Scrollbar(main_frame, orient=tk.VERTICAL, command=canvas.yview)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    canvas.configure(yscrollcommand=scrollbar.set)
    canvas.bind('<Configure>', lambda e: canvas.configure(scrollregion=canvas.bbox("all")))

    def _on_mousewheel(event): # Mausrad-Scrolling
        delta = 0;
        if event.num == 4: delta = -1
        elif event.num == 5: delta = 1
        elif hasattr(event, 'delta'): delta = -event.delta / 120
        if delta != 0: canvas.yview_scroll(int(delta), "units")
    canvas.bind("<MouseWheel>", _on_mousewheel); canvas.bind("<Button-4>", _on_mousewheel); canvas.bind("<Button-5>", _on_mousewheel)

    scrollable_frame = ttk.Frame(canvas, padding="10")
    canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")

    # --- Parameter Container ---
    param_container = ttk.LabelFrame(scrollable_frame, text="Simulationseinstellungen", padding="10")
    param_container.grid(row=0, column=0, sticky=tk.NSEW, padx=10, pady=10)
    param_container.columnconfigure(1, weight=1)

    # Parameter-Eingabefelder...
    ttk.Label(param_container, text="Lernrate:").grid(row=0, column=0, sticky=tk.W, pady=3)
    learning_rate_entry = ttk.Entry(param_container, width=10); learning_rate_entry.insert(0, str(DEFAULT_LEARNING_RATE)); learning_rate_entry.grid(row=0, column=1, sticky=tk.EW, pady=3, padx=5); entry_widgets['learning_rate'] = learning_rate_entry
    ttk.Label(param_container, text="Decay Rate:").grid(row=1, column=0, sticky=tk.W, pady=3)
    decay_rate_entry = ttk.Entry(param_container, width=10); decay_rate_entry.insert(0, str(DEFAULT_DECAY_RATE)); decay_rate_entry.grid(row=1, column=1, sticky=tk.EW, pady=3, padx=5); entry_widgets['decay_rate'] = decay_rate_entry
    ttk.Label(param_container, text="Reward Interval:").grid(row=2, column=0, sticky=tk.W, pady=3)
    reward_interval_entry = ttk.Entry(param_container, width=10); reward_interval_entry.insert(0, str(DEFAULT_REWARD_INTERVAL)); reward_interval_entry.grid(row=2, column=1, sticky=tk.EW, pady=3, padx=5); entry_widgets['reward_interval'] = reward_interval_entry
    ttk.Label(param_container, text="Epochen:").grid(row=3, column=0, sticky=tk.W, pady=3)
    epochs_entry = ttk.Entry(param_container, width=10); epochs_entry.insert(0, str(DEFAULT_EPOCHS)); epochs_entry.grid(row=3, column=1, sticky=tk.EW, pady=3, padx=5); entry_widgets['epochs'] = epochs_entry

    # --- ### NEU: Gemini Prompt Eingabefeld ### ---
    ttk.Label(param_container, text="Analyse-Prompt für Gemini:").grid(row=4, column=0, columnspan=2, sticky=tk.W, pady=(10, 2))
    gemini_prompt_text_widget = tk.Text(param_container, height=4, width=40, wrap=tk.WORD, relief=tk.SOLID, borderwidth=1)
    gemini_prompt_text_widget.grid(row=5, column=0, columnspan=2, sticky=tk.EW, pady=2, padx=5)
    # Füge einen Platzhaltertext oder eine Anweisung hinzu
    gemini_prompt_text_widget.insert("1.0", "Optional: Geben Sie hier Ihre spezifische Analysefrage oder den Fokus für den Gemini-Bericht ein (z.B. 'Fokus auf Wachstumspotenzial', 'Risikoanalyse', 'Vergleich mit Branche').")
    # Optional: Verhalten bei Klick/Fokus, um Platzhalter zu löschen
    def clear_placeholder(event):
        if gemini_prompt_text_widget.get("1.0", tk.END).strip() == "Optional: Geben Sie hier Ihre spezifische Analysefrage oder den Fokus für den Gemini-Bericht ein (z.B. 'Fokus auf Wachstumspotenzial', 'Risikoanalyse', 'Vergleich mit Branche').":
            gemini_prompt_text_widget.delete("1.0", tk.END)
            gemini_prompt_text_widget.config(fg='black') # Normale Textfarbe
    def add_placeholder(event):
        if not gemini_prompt_text_widget.get("1.0", tk.END).strip():
            gemini_prompt_text_widget.insert("1.0", "Optional: Geben Sie hier Ihre spezifische Analysefrage oder den Fokus für den Gemini-Bericht ein (z.B. 'Fokus auf Wachstumspotenzial', 'Risikoanalyse', 'Vergleich mit Branche').")
            gemini_prompt_text_widget.config(fg='grey') # Platzhalterfarbe
    gemini_prompt_text_widget.config(fg='grey')
    gemini_prompt_text_widget.bind("<FocusIn>", clear_placeholder)
    gemini_prompt_text_widget.bind("<FocusOut>", add_placeholder)


    # --- Buttons ---
    button_frame = ttk.Frame(param_container)
    button_frame.grid(row=6, column=0, columnspan=2, pady=(20, 10), sticky=tk.EW) # Angepasste Reihe
    start_button = ttk.Button(button_frame, text="Simulation starten", command=lambda: start_simulation_action(root, status_label))
    start_button.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
    save_button = ttk.Button(button_frame, text="Speichern", command=lambda: save_gui_settings(status_label))
    save_button.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
    load_button = ttk.Button(button_frame, text="Laden", command=lambda: load_gui_settings(status_label))
    load_button.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)

    # --- Status Label ---
    status_label = ttk.Label(param_container, text="Status: Bereit", anchor=tk.W)
    status_label.grid(row=7, column=0, columnspan=2, pady=(5, 10), sticky=tk.EW) # Angepasste Reihe

    # --- Gewichte Container ---
    # ... (unverändert) ...
    weights_container = ttk.LabelFrame(scrollable_frame, text="Modul-Kategorie Gewichte", padding="10")
    weights_container.grid(row=0, column=1, sticky=tk.NSEW, padx=10, pady=10)
    module_labels_gui = ["Cortex Creativus", "Simulatrix Neuralis", "Cortex Criticus", "Limbus Affektus", "Meta Cognitio", "Cortex Socialis"]; category_labels_gui = data['Kategorie'].unique() if data is not None else []
    weight_sliders.clear(); slider_row = 0; weights_container.columnconfigure(1, weight=1)
    for module_label in module_labels_gui:
        module_header = ttk.Label(weights_container, text=f"{module_label}:", font=("Helvetica", 11, "bold")); module_header.grid(row=slider_row, column=0, columnspan=2, sticky=tk.W, pady=(10, 2)); slider_row += 1
        for category_label in category_labels_gui:
            cat_label_widget = ttk.Label(weights_container, text=f"  → {category_label}:"); cat_label_widget.grid(row=slider_row, column=0, sticky=tk.W, padx=5, pady=1)
            weight_slider = ttk.Scale(weights_container, from_=0.0, to=1.0, orient=tk.HORIZONTAL, value=DEFAULT_MODULE_CATEGORY_WEIGHT, length=200); weight_slider.set(DEFAULT_MODULE_CATEGORY_WEIGHT); weight_slider.grid(row=slider_row, column=1, sticky=tk.EW, padx=5, pady=1)
            weight_sliders[(module_label, category_label)] = weight_slider; slider_row += 1

    scrollable_frame.columnconfigure(0, weight=1); scrollable_frame.columnconfigure(1, weight=2); scrollable_frame.rowconfigure(0, weight=1)

    # --- Save/Load Settings Funktionen ---
    # ... (save_gui_settings, load_gui_settings unverändert) ...
    def save_gui_settings(status_label_ref):
        global entry_widgets, weight_sliders; settings_data = {"basic_params": {},"module_weights": {}}
        try:
            for name, widget in entry_widgets.items(): settings_data["basic_params"][name] = widget.get()
            for (module, category), slider in weight_sliders.items():
                if module not in settings_data["module_weights"]: settings_data["module_weights"][module] = {}
                settings_data["module_weights"][module][category] = slider.get()
            filepath = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("JSON files", "*.json"), ("All files", "*.*")], initialfile=SETTINGS_FILENAME, title="GUI-Einstellungen speichern")
            if not filepath: status_label_ref.config(text="Status: Speichern abgebrochen."); return
            with open(filepath, 'w', encoding='utf-8') as f: json.dump(settings_data, f, indent=4, ensure_ascii=False)
            status_label_ref.config(text=f"Status: Einstellungen gespeichert in {os.path.basename(filepath)}"); print(f"GUI-Einstellungen gespeichert in {filepath}")
        except Exception as e: messagebox.showerror("Fehler Speichern", f"Konnte nicht speichern:\n{e}"); status_label_ref.config(text="Status: Fehler Speichern."); print(f"FEHLER Speichern GUI-Einstellungen: {e}")

    def load_gui_settings(status_label_ref):
        global entry_widgets, weight_sliders
        try:
            filepath = filedialog.askopenfilename(defaultextension=".json", filetypes=[("JSON files", "*.json"), ("All files", "*.*")], initialfile=SETTINGS_FILENAME, title="GUI-Einstellungen laden")
            if not filepath: status_label_ref.config(text="Status: Laden abgebrochen."); return
            if not os.path.exists(filepath): messagebox.showerror("Fehler", f"Datei nicht gefunden: {filepath}"); status_label_ref.config(text="Status: Ladedatei fehlt."); return
            with open(filepath, 'r', encoding='utf-8') as f: settings_data = json.load(f)
            loaded_params = settings_data.get("basic_params", {});
            for name, widget in entry_widgets.items():
                if name in loaded_params: widget.delete(0, tk.END); widget.insert(0, str(loaded_params[name]))
                else: print(f"Warnung: Parameter '{name}' nicht in Datei.")
            loaded_weights = settings_data.get("module_weights", {}); warnings = []; missing_keys = []
            for module, categories in loaded_weights.items():
                for category, weight in categories.items():
                    key = (module, category)
                    if key in weight_sliders:
                        try: weight_sliders[key].set(float(weight))
                        except (ValueError, TypeError): warnings.append(f"Wert '{weight}' für '{module}->{category}' ungültig")
                    else: missing_keys.append(f"'{module}->{category}'")
            if warnings: print(f"Warnung: Ungültige Gewichtswerte: {', '.join(warnings)}")
            if missing_keys: warn_msg = f"Einstellungen passen nicht zu Modulen/Kat.:\n {', '.join(missing_keys)}"; print(f"Warnung: {warn_msg}"); messagebox.showwarning("Laden (Warnung)", warn_msg)
            status_label_ref.config(text=f"Status: Geladen aus {os.path.basename(filepath)}"); print(f"GUI-Einstellungen geladen aus {filepath}")
        except json.JSONDecodeError as e: messagebox.showerror("Fehler Laden", f"Ungültiges JSON:\n{e}"); status_label_ref.config(text="Status: Laden fehlgeschlagen (JSON)."); print(f"FEHLER Laden GUI-Einstellungen (JSON): {e}")
        except Exception as e: messagebox.showerror("Fehler Laden", f"Konnte nicht laden:\n{e}"); status_label_ref.config(text="Status: Laden fehlgeschlagen."); print(f"FEHLER Laden GUI-Einstellungen: {e}")


    # --- Simulation Start Action (angepasst) ---
    def start_simulation_action(gui_root, status_label_ref):
        # Validierung...
        try:
            lr = float(learning_rate_entry.get().replace(',', '.')); dr = float(decay_rate_entry.get().replace(',', '.')); ri = int(reward_interval_entry.get()); ep = int(epochs_entry.get())
            if not (0 < lr <= 1.0): raise ValueError("Lernrate >0 und <=1");
            if not (0 <= dr < 1.0): raise ValueError("Decay Rate >=0 und <1");
            if not (ri >= 1): raise ValueError("Reward Interval >= 1");
            if not (ep >= 1): raise ValueError("Epochen >= 1");
        except ValueError as ve: messagebox.showerror("Eingabefehler", f"Ungültiger Wert: {ve}."); status_label_ref.config(text="Status: Fehler."); return
        except Exception as e: messagebox.showerror("Fehler", f"Fehler Eingabe: {e}"); status_label_ref.config(text=f"Status: Fehler - {e}"); return

        # --- ### NEU: Gemini Prompt holen ### ---
        user_gemini_prompt = gemini_prompt_text_widget.get("1.0", tk.END).strip()
        # Entferne Platzhaltertext, falls noch vorhanden
        placeholder = "Optional: Geben Sie hier Ihre spezifische Analysefrage oder den Fokus für den Gemini-Bericht ein (z.B. 'Fokus auf Wachstumspotenzial', 'Risikoanalyse', 'Vergleich mit Branche')."
        if user_gemini_prompt == placeholder:
            user_gemini_prompt = "" # Leerer String, wenn nur Platzhalter da war

        weights = {key: slider.get() for key, slider in weight_sliders.items()}
        status_label_ref.config(text="Status: Simulation läuft...")

        # Starte Simulation im Thread, übergib auch den Gemini-Prompt
        threading.Thread(target=run_simulation_with_callback,
                            args=(lr, dr, ri, ep, gui_root, weights, status_label_ref, user_gemini_prompt), # ### NEU: Prompt übergeben ###
                            daemon=True).start()


    # --- Simulation Callback (angepasst für Gemini) ---
    def run_simulation_with_callback(lr, dr, ri, ep, gui_root, weights, status_label_ref, user_gemini_prompt): # ### NEU: Prompt empfangen ###
        gemini_report = None # Initialisiere Gemini-Bericht
        try:
             # Führe die Simulation aus
             simulation_results = run_simulation_from_gui(lr, dr, ri, ep, gui_root, weights)

             # Prüfe, ob Simulation erfolgreich war und Ergebnisse lieferte
             if simulation_results is None:
                 if gui_root.winfo_exists():
                     gui_root.after(0, lambda: status_label_ref.config(text="Status: Simulation fehlgeschlagen."))
                 return # Breche hier ab

             # Entpacke Ergebnisse (jetzt 4 Elemente)
             activation_history, weights_history, module_labels, final_report_text = simulation_results
             plot_data = (activation_history, weights_history, module_labels) # Tupel für Plot-Funktion

             # --- ### NEU: Gemini Analyse aufrufen (falls Prompt vorhanden) ### ---
             if user_gemini_prompt and gemini_available:
                 if gui_root.winfo_exists():
                     gui_root.after(0, lambda: status_label_ref.config(text="Status: Rufe Gemini API auf..."))
                 gemini_report = get_gemini_report(user_gemini_prompt, final_report_text) # Rufe API auf
                 if gui_root.winfo_exists():
                     # Update Status erst nach API-Aufruf
                     if "Fehler:" in gemini_report:
                         gui_root.after(0, lambda: status_label_ref.config(text="Status: Gemini API Fehler."))
                     else:
                         gui_root.after(0, lambda: status_label_ref.config(text="Status: Gemini-Analyse abgeschlossen."))

             # --- Plots und finale Meldungen im GUI-Thread planen ---
             if gui_root.winfo_exists():
                 # Plane zuerst Plots
                 if plot_data and (plot_data[0] or plot_data[1]): # Nur wenn es Plotdaten gibt
                     gui_root.after(10, lambda: generate_and_display_plots(plot_data, gui_root, status_label_ref)) # Kleine Verzögerung für Status-Update
                 else:
                     # Wenn keine Plots, aber Simulation OK, aktualisiere Status
                     gui_root.after(10, lambda: status_label_ref.config(text="Status: Simulation abgeschlossen (keine Plots)."))
                     gui_root.after(10, lambda: messagebox.showinfo("Abgeschlossen", f"Simulation beendet.\nBericht/HTML erstellt."))

                 # Plane Anzeige des Gemini-Berichts danach (falls vorhanden)
                 if gemini_report:
                     # Nutze 'after', um sicherzustellen, dass es im GUI-Thread läuft
                     gui_root.after(20, lambda rep=gemini_report: display_gemini_report(rep, gui_root)) # Übergib Bericht an Lambda

        except Exception as e:
             # Fange Fehler ab, die direkt in der Simulation auftreten
             print(f"FEHLER während der Simulation im Thread: {e}")
             if gui_root.winfo_exists():
                  gui_root.after(0, lambda: status_label_ref.config(text=f"Status: Simulationsfehler!"))
                  gui_root.after(0, lambda: messagebox.showerror("Simulationsfehler", f"Fehler während der Simulation:\n{e}"))

    # Lade initiale Einstellungen (falls vorhanden)
    if os.path.exists(SETTINGS_FILENAME):
        load_gui_settings(status_label)
    else: print(f"Keine Einstellungsdatei '{SETTINGS_FILENAME}' gefunden, verwende Standardwerte.")

    # Prüfe API Key beim Start (optional, aber hilfreich)
    if gemini_available and not os.getenv("GEMINI_API_KEY"):
        messagebox.showwarning("Gemini Konfiguration", "Kein Gemini API Key in Umgebungsvariable 'GEMINI_API_KEY' gefunden.\nDie Gemini-Analyse wird nicht funktionieren.")

    root.mainloop()

# --- Programmstart ---
if __name__ == "__main__":
    main()