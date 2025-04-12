# -*- coding: utf-8 -*-
# Filename: neuropersona_core.py
# --- Imports ---
import pandas as pd
import numpy as np
import random
import matplotlib
matplotlib.use('TkAgg') # TkAgg Backend für Kompatibilität mit Tkinter
import matplotlib.pyplot as plt
import math # Für komplexere Berechnungen
from collections import Counter, deque # deque für History
import json
import importlib # Für dynamischen Import in GUI
import sqlite3 # Für persistentes Gedächtnis
import os
import time
import threading
from typing import Optional, Callable, List, Tuple, Dict

import tkinter as tk
from tkinter import ttk, messagebox, filedialog, scrolledtext
entry_widgets = {}

# Optional: networkx für Graphenplot
try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False
    print("Warnung: networkx nicht gefunden. Netzwerk-Graph-Plot wird nicht erstellt.")

# tqdm für Fortschrittsbalken
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    print("Warnung: tqdm nicht gefunden. Fortschrittsbalken in Konsole nicht verfügbar.")
    # Fallback-Funktion, falls tqdm nicht verfügbar ist
    def tqdm(iterable, *args, **kwargs):
        return iterable


# --- Globale Instanzen / Zustände ---
CURRENT_EMOTION_STATE = {} # Wird in initialize_network_nodes gesetzt
persistent_memory_manager = None # Wird global initialisiert
activation_history: Dict[str, deque] = {} # Global für Limbus Zugriff, wird in Sim. initialisiert

# --- Konstanten ---
# Kern
MODEL_FILENAME = "neuropersona_enhanced_state_v_dare_complete.json"
SETTINGS_FILENAME = "neuropersona_gui_settings.json"
PLOTS_FOLDER = "plots"
DEFAULT_EPOCHS = 30
DEFAULT_LEARNING_RATE = 0.025
DEFAULT_DECAY_RATE = 0.012
DEFAULT_REWARD_INTERVAL = 5
DEFAULT_ACTIVATION_THRESHOLD_PROMOTION = 0.7
DEFAULT_HISTORY_LENGTH_MAP_PROMOTION = {"short_term": 5, "mid_term": 20}
DEFAULT_MODULE_CATEGORY_WEIGHT = 0.15
HISTORY_MAXLEN = 150

# Emotionen (PAD)
EMOTION_DIMENSIONS = ["pleasure", "arousal", "dominance"]
INITIAL_EMOTION_STATE = {dim: 0.0 for dim in EMOTION_DIMENSIONS}
EMOTION_UPDATE_RATE = 0.025
EMOTION_VOLATILITY = 0.0255
EMOTION_DECAY_TO_NEUTRAL = 0.05

# Werte
DEFAULT_VALUES = {"Innovation": 0.1, "Sicherheit": 0.1, "Effizienz": 0.1, "Ethik": 0.1, "Neugier": 0.1}
VALUE_UPDATE_RATE = 0.15
VALUE_INFLUENCE_FACTOR = 0.15

# Verstärkung
REINFORCEMENT_PLEASURE_THRESHOLD = 0.3
REINFORCEMENT_CRITIC_THRESHOLD = 0.7
REINFORCEMENT_FACTOR = 0.015

# Persistenz
PERSISTENT_MEMORY_DB = "neuropersona_longterm_memory_v_dare_complete.db"
PERSISTENT_MEMORY_TABLE = "core_memories"
PERSISTENT_REFLECTION_TABLE = "reflection_log"
MEMORY_RELEVANCE_THRESHOLD = 0.6
MEMORY_CONSOLIDATION_INTERVAL = 15

# Plastizität
STRUCTURAL_PLASTICITY_INTERVAL = 8
PRUNING_THRESHOLD = 0.01
SPROUTING_THRESHOLD = 0.80
SPROUTING_NEW_WEIGHT_MEAN = 0.03
MAX_CONNECTIONS_PER_NODE = 60
NODE_PRUNING_ENABLED = False # SICHERHEITSSCHALTER!
NODE_INACTIVITY_THRESHOLD_EPOCHS = 40
NODE_INACTIVITY_ACTIVATION = 0.05

# Meta-Kognition
REFLECTION_LOG_MAX_LEN = 200
STAGNATION_DETECTION_WINDOW = 7
STAGNATION_THRESHOLD = 0.005
OSCILLATION_DETECTION_WINDOW = 10
OSCILLATION_THRESHOLD_STD = 0.3


def random_neuron_type() -> str:
    r = random.random(); return "excitatory" if r < 0.7 else ("inhibitory" if r < 0.95 else "interneuron")


def sigmoid(x):
    with np.errstate(over='ignore', under='ignore'): return 1 / (1 + np.exp(-np.clip(x, -700, 700)))

def add_activation_noise(activation, noise_level=0.05):
    arousal = CURRENT_EMOTION_STATE.get('arousal', 0.0) if CURRENT_EMOTION_STATE else 0.0
    arousal_factor = 1.0 + abs(arousal)
    noise = np.random.normal(0, noise_level * arousal_factor)
    return np.clip(activation + noise, 0.0, 1.0)

def calculate_dynamic_learning_rate(base_lr, emotion_state, meta_cognitive_state):
    arousal = emotion_state.get('arousal', 0.0); pleasure = emotion_state.get('pleasure', 0.0)
    emo_factor = 1.0 + (arousal * 0.3) + (pleasure * 0.2); meta_factor = meta_cognitive_state.get('lr_boost', 1.0)
    return np.clip(base_lr * emo_factor * meta_factor, 0.001, 0.6)

def _default_status_callback(message: str): print(f"[Status] {message}")


def create_html_report(final_summary: str, final_recommendation: str, interpretation_log: list, important_categories: list, structured_results: dict, plots_folder: str = PLOTS_FOLDER, output_html: str = "neuropersona_report.html") -> None:
    if not os.path.exists(plots_folder): os.makedirs(plots_folder, exist_ok=True)
    try:
        all_files = [f for f in os.listdir(plots_folder) if f.endswith(".png")]
        # Definierte Reihenfolge der Plots
        plot_order = [
            "plot_act_weights.png", "plot_dynamics.png", "plot_modules.png",
            "plot_emo_values.png", "plot_structure_stats.png", "plot_network_graph.png"
        ]
        # Sortiere gefundene Plots gemäß der Reihenfolge, andere alphabetisch danach
        plots_in_order = [p for p in plot_order if p in all_files]
        other_plots = sorted([f for f in all_files if f not in plot_order])
        plots = plots_in_order + other_plots

    except FileNotFoundError: plots = []
    recommendation_color = {"Empfehlung": "#28a745", "Empfehlung (moderat)": "#90ee90", "Abwarten": "#ffc107", "Abwarten (Instabil/Schwach)": "#ffe066", "Abraten": "#dc3545", "Abraten (moderat)": "#f08080"}.get(final_recommendation, "#6c757d")
    final_emotion = structured_results.get('emotion_state', {}); final_values = structured_results.get('value_node_activations', {}); reflections = structured_results.get('reflection_summary', [])
    exec_time = structured_results.get('execution_time_seconds', None)
    stability = structured_results.get('stability_assessment', 'N/A')

    try:
        with open(output_html, "w", encoding="utf-8") as f:
            f.write("""<!DOCTYPE html><html lang='de'><head><meta charset='UTF-8'><title>NeuroPersona Analysebericht (Erweitert)</title><style>body{font-family:Arial,sans-serif;margin:20px;background-color:#f8f9fa;color:#212529}h1,h2,h3{color:#343a40; border-bottom: 1px solid #dee2e6; padding-bottom: 5px;} .report-container{max-width: 900px; margin: auto; background: #fff; padding: 30px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,.1);} .prognosis{background:""" + recommendation_color + """;color:white;padding:15px;border-radius:8px;font-size:1.15em; margin-bottom: 20px; text-align: center;} details{margin-top:15px;background:#f8f9fa;border:1px solid #dee2e6;border-radius:5px;padding:10px;} summary{font-weight:700;cursor:pointer;color:#0056b3;} summary:hover{color:#003d80;} img{max-width:95%;height:auto;margin-top:10px;border:1px solid #dee2e6;border-radius:5px;display:block;margin-left:auto; margin-right:auto;} pre{white-space: pre-wrap; word-wrap: break-word; background: #e9ecef; padding: 10px; border-radius: 4px; font-size: 0.9em;}.footer{margin-top:40px;text-align:center;font-size:.85em;color:#adb5bd}.epoch-entry{border-bottom: 1px dashed #eee; padding-bottom: 5px; margin-bottom: 5px; font-size: 0.9em;} .epoch-entry:last-child{border-bottom: none;} .category-list li{margin-bottom: 3px;}</style></head><body><div class='report-container'>""")
            f.write(f"<h1>NeuroPersona Analysebericht</h1>")
            f.write(f"<div class='prognosis'>📈 <b>Gesamteinschätzung: {final_recommendation}</b></div>")

            f.write("<details open><summary>📋 Kernpunkte & Netzwerkzustand</summary>")
            f.write(f"<h3>Dominante Tendenz</h3>")
            dom_cat = structured_results.get('dominant_category', 'N/A')
            dom_act = structured_results.get('dominant_activation', 0.0)
            f_dom_cat = structured_results.get('frequent_dominant_category', 'N/A')
            f.write(f"<p><b>Finale Dominanz:</b> {dom_cat} ({dom_act:.3f})<br>")
            f.write(f"<b>Häufigste Dominanz im Verlauf:</b> {f_dom_cat}<br>")
            f.write(f"<b>Stabilität der Dominanz:</b> {stability}</p>")

            f.write(f"<h3>Wichtigste Kategorien (Final)</h3>")
            if important_categories:
                f.write("<ul class='category-list'>")
                for cat, imp in important_categories: f.write(f"<li><b>{cat}</b>: {imp}</li>")
                f.write("</ul>")
            else: f.write("<p>Keine hervorstechenden Kategorien identifiziert.</p>")
            f.write("<br><i>Detaillierte Zusammenfassung:</i><pre>" + final_summary + "</pre>") # Original Summary Text
            f.write("</details>")


            f.write("<details><summary>🧠 Kognitiver & Emotionaler Zustand (Final)</summary>")
            f.write("<h3>Emotionale Grundstimmung (PAD):</h3><pre>" + json.dumps(final_emotion, indent=2) + "</pre>")
            f.write("<h3>Aktive Wertvorstellungen:</h3><pre>" + json.dumps(final_values, indent=2) + "</pre>")
            f.write("</details>")

            if reflections:
                f.write("<details><summary>🤔 Meta-Kognitive Reflexionen (Letzte 5 Einträge)</summary>")
                for entry in reflections[:5]: # Zeige nur die letzten 5 für Übersicht
                     msg = entry.get('message','(Keine Nachricht)'); data_str = json.dumps(entry.get('data',{})) if entry.get('data') else ""; f.write(f"<p><b>E{entry.get('epoch','?')}:</b> {msg} {f'<small><i>({data_str})</i></small>' if data_str else ''}</p>")
                f.write("</details>")

            if interpretation_log:
                f.write("<details><summary>📈 Analyseverlauf (Auszug letzter Epochen)</summary>")
                log_subset = interpretation_log[-10:] # Zeige nur die letzten 10 Epochen
                for entry in reversed(log_subset): # Neueste zuerst
                    epoch = entry.get('epoch', '-')
                    dom = entry.get('dominant_category', '-')
                    act = entry.get('dominant_activation', 0.0)
                    emo_pleasure = entry.get('emotion_state', {}).get('pleasure', 0.0)
                    val_innov = entry.get('value_node_activations', {}).get('Innovation', 0.0)
                    act_val = float(act) if isinstance(act, (float, int, np.number)) and not np.isnan(act) else 0.0
                    act_str = f"{act_val:.2f}"
                    f.write(f"<div class='epoch-entry'><b>E{epoch}:</b> Dom: {dom} ({act_str}), P: {emo_pleasure:.2f}, Innov: {val_innov:.2f}</div>")
                if len(interpretation_log) > 10: f.write("<p><i>... (ältere Epochen nicht angezeigt)</i></p>")
                f.write("</details>")

            if plots:
                f.write("<details open><summary>🖼️ Visualisierungen</summary>"); [f.write(f"<p style='text-align:center; font-weight:bold; margin-top:15px;'>{plot.replace('.png','').replace('_',' ').title()}</p><img src='{plots_folder}/{plot}' alt='{plot}'><br>") for plot in plots]; f.write("</details>")
            else: f.write("<details><summary>🖼️ Visualisierungen</summary><p>Keine Plots gefunden oder generiert.</p></details>")

            f.write("<div class='footer'>")
            if exec_time: f.write(f"Analyse durchgeführt in {exec_time:.2f} Sekunden. ")
            f.write(f"Erstellt mit NeuroPersona KI-System (v_dare_complete) am {time.strftime('%d.%m.%Y %H:%M:%S')}</div>")
            f.write("</div></body></html>") # Container schließen
        print(f"✅ Erweiterten HTML-Report erstellt: {output_html}")
    except Exception as e: print(f"FEHLER beim Schreiben des erweiterten HTML-Reports '{output_html}': {e}")


# --- Netzwerk-Hilfsfunktionen (Decay, Reward, Social, Context, Hebb) ---
def decay_weights(nodes_list, decay_rate=0.002, forgetting_curve=0.98):
    factor = np.clip((1 - decay_rate) * forgetting_curve, 0.0, 1.0)
    for node in nodes_list:
        if hasattr(node, 'connections'):
            for conn in node.connections: conn.weight *= factor

def reward_connections(nodes_list, target_label, reward_factor=0.05):
    # Hinweis: Diese Funktion wird aktuell in simulate_learning_cycle NICHT aufgerufen.
    # apply_reinforcement wird stattdessen verwendet.
    for node in nodes_list:
         if hasattr(node, 'connections'):
            for conn in node.connections:
                if hasattr(conn.target_node, 'label') and conn.target_node.label == target_label:
                    conn.weight += reward_factor * getattr(node, 'activation', 0.0)
                    conn.weight = np.clip(conn.weight, 0.0, 1.0)
def convert_text_answer_to_numeric(answer_text: str) -> float:
    """Konvertiert Textantwort in normierten numerischen Wert (0.0 - 1.0)."""
    if not isinstance(answer_text, str): return 0.5
    text = answer_text.strip().lower()
    mapping = {
        "sehr hoch": 0.95, "hoch": 0.8, "eher hoch": 0.7, "mittel": 0.5, "eher niedrig": 0.35, "niedrig": 0.3, "gering": 0.2, "sehr gering": 0.05,
        "ja": 0.9, "eher ja": 0.7, "nein": 0.1, "eher nein": 0.3,
        "positiv": 0.85, "eher positiv": 0.65, "negativ": 0.15, "eher negativ": 0.35,
        "neutral": 0.5, "unsicher": 0.4, "sicher": 0.9, "unbekannt": 0.5,
        "stimme voll zu": 1.0, "stimme zu": 0.8, "stimme eher zu": 0.6,
        "lehne ab": 0.2, "lehne voll ab": 0.0, "lehne eher ab": 0.4,
        "trifft voll zu": 1.0, "trifft zu": 0.8, "trifft eher zu": 0.6,
        "trifft nicht zu": 0.2, "trifft gar nicht zu": 0.0, "trifft eher nicht zu": 0.4
    }
    if text in mapping: return mapping[text]
    # Heuristiken
    if "hoch" in text or "stark" in text or "viel" in text: return 0.8
    if "gering" in text or "niedrig" in text or "wenig" in text: return 0.2
    if "mittel" in text or "moderat" in text: return 0.5
    if "positiv" in text or "chance" in text or "gut" in text: return 0.85
    if "negativ" in text or "risiko" in text or "schlecht" in text: return 0.15
    if "ja" in text or "zustimm" in text or "trifft zu" in text: return 0.9
    if "nein" in text or "ablehn" in text or "trifft nicht zu" in text: return 0.1
    return 0.5 # Default: Neutral

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
    data['Frage'] = data['Frage'].astype(str).str.strip()
    data['Kategorie'] = data['Kategorie'].astype(str).str.strip()
    # Fülle leere Kategorien *vor* der Normalisierung
    data['Kategorie'] = data['Kategorie'].replace('', 'Unkategorisiert')
    data['Kategorie'] = data['Kategorie'].fillna('Unkategorisiert')

    data['normalized_answer'] = data['Antwort'].apply(convert_text_answer_to_numeric)

    # Überprüfe nochmal nach der Normalisierung, ob leere Kategorien übrig sind (sollte nicht sein)
    if (data['Kategorie'] == '').any():
        print("WARNUNG: Nach Verarbeitung immer noch leere Kategorien. Ersetze durch 'Unbekannt'.")
        data['Kategorie'].replace('', 'Unbekannt', inplace=True)

    print(f"Datenvorverarbeitung abgeschlossen. {len(data)} Zeilen verarbeitet. Kategorien: {data['Kategorie'].nunique()}")
    # print(f"Beispiel Normalisierung: '{data['Antwort'].iloc[0]}' -> {data['normalized_answer'].iloc[0]}") # Optional: Debug Output
    return data

def social_influence(nodes_list, social_network, influence_factor=0.05):
    """Modifiziert Verbindungsgewichte basierend auf externen sozialen Faktoren."""
    applied_count = 0
    for node in nodes_list:
        if not hasattr(node, 'label') or node.label not in social_network: continue
        social_impact = social_network.get(node.label, 0.0) * influence_factor
        if social_impact <= 0.001: continue
        # Finde eingehende Verbindungen zu diesem Knoten
        for source_node in nodes_list:
            if hasattr(source_node, 'connections'):
                for conn in source_node.connections:
                    if hasattr(conn.target_node, 'label') and conn.target_node.label == node.label:
                        reinforcement = social_impact * getattr(source_node, 'activation', 0.5)
                        conn.weight = np.clip(conn.weight + reinforcement, 0, 1.0)
                        applied_count += 1
    # if applied_count > 0: print(f"[Social Influence] {applied_count} connections adjusted.")

def apply_contextual_factors(activation, node, context_factors):
    """Wendet kontextuelle Faktoren auf die Aktivierung an (aktuell nicht genutzt)."""
    if not hasattr(node, 'label') or not context_factors: return activation
    context_factor = context_factors.get(node.label, 1.0) * random.uniform(0.95, 1.05) # Leichte Variation
    return np.clip(activation * context_factor, 0.0, 1.0)

def hebbian_learning(node, learning_rate=0.1, weight_limit=1.0, reg_factor=0.001):
    """Wendet Hebb'sches Lernen auf die ausgehenden Verbindungen eines Knotens an."""
    if not hasattr(node, 'connections') or not hasattr(node, 'activation') or node.activation < 0.05: return
    node_act = node.activation
    for conn in node.connections:
        target_act = getattr(conn.target_node, 'activation', 0.0)
        if target_act < 0.05: continue # Nur lernen, wenn beide aktiv sind
        delta_weight = learning_rate * node_act * target_act # Hebb'sche Regel
        conn.weight += delta_weight
        conn.weight -= reg_factor * conn.weight # Regularisierung/Decay
        conn.weight = np.clip(conn.weight, 0.0, weight_limit)

# --- Plastizitätsfunktionen  ---
def prune_connections(nodes_list: list, threshold: float = PRUNING_THRESHOLD):
    pruned_count = 0
    for node in nodes_list:
        if hasattr(node, 'connections'):
            original_count = len(node.connections)
            node.connections = [conn for conn in node.connections if conn.weight >= threshold]
            pruned_count += original_count - len(node.connections)
    # if pruned_count > 0: print(f"[Plasticity] Pruned {pruned_count} weak connections.")
    return pruned_count

def sprout_connections(nodes_list: list, activation_history_local: dict, threshold: float = SPROUTING_THRESHOLD, max_conns: int = MAX_CONNECTIONS_PER_NODE, new_weight_mean: float = SPROUTING_NEW_WEIGHT_MEAN):
    sprouted_count = 0
    node_map = {node.label: node for node in nodes_list if hasattr(node, 'label')}
    # Nimm die letzte Aktivierung aus der übergebenen History
    last_activations = {label: history[-1] for label, history in activation_history_local.items() if history}
    # Finde Knoten über dem Sprouting-Schwellwert
    active_nodes = [label for label, act in last_activations.items() if act > threshold]
    if len(active_nodes) < 2: return 0

    # Mische die aktiven Knoten, um zufällige Paarungen zu fördern
    random.shuffle(active_nodes)

    # Versuche, eine begrenzte Anzahl neuer Verbindungen zu erstellen
    max_sprouts_per_epoch = max(1, int(len(active_nodes) * 0.1)) # Z.B. 10% der aktiven Knoten
    attempted_sprouts = 0

    for i, label1 in enumerate(active_nodes):
        if sprouted_count >= max_sprouts_per_epoch: break
        node1 = node_map.get(label1)
        if not node1 or not hasattr(node1, 'connections') or len(node1.connections) >= max_conns: continue

        # Suche nach einem zufälligen, *anderen* aktiven Knoten für eine Verbindung
        potential_partners = active_nodes[i+1:] + active_nodes[:i] # Alle anderen Knoten
        random.shuffle(potential_partners)

        for label2 in potential_partners:
            if attempted_sprouts > len(active_nodes) * 2: break # Verhindere Endlosschleife bei dichten Graphen
            attempted_sprouts += 1

            node2 = node_map.get(label2)
            if not node2 or not hasattr(node2, 'connections') or len(node2.connections) >= max_conns: continue

            # Prüfe, ob Verbindung (in beide Richtungen) bereits existiert
            conn_exists_12 = any(conn.target_node == node2 for conn in node1.connections)
            conn_exists_21 = any(conn.target_node == node1 for conn in node2.connections)

            if not conn_exists_12 and not conn_exists_21:
                new_weight = max(0.001, np.random.normal(new_weight_mean, new_weight_mean / 2))
                node1.add_connection(node2, weight=new_weight)
                sprouted_count += 1
                # Optional: Bidirektionale Verbindung
                # if len(node2.connections) < max_conns: node2.add_connection(node1, weight=new_weight)
                break # Nimm den nächsten Knoten label1, nachdem eine Verbindung für ihn erstellt wurde

    # if sprouted_count > 0: print(f"[Plasticity] Sprouted {sprouted_count} new connections.")
    return sprouted_count

def prune_inactive_nodes(nodes_list: list, activation_history_local: dict, current_epoch: int, threshold_epochs: int = NODE_INACTIVITY_THRESHOLD_EPOCHS, activation_threshold: float = NODE_INACTIVITY_ACTIVATION, enabled: bool = NODE_PRUNING_ENABLED):
    if not enabled: return nodes_list, 0
    nodes_to_remove_labels = set()
    # Definiere Klassen, die *nicht* gepruned werden sollen
    protected_classes = (CortexCreativus, SimulatrixNeuralis, CortexCriticus, LimbusAffektus, MetaCognitio, CortexSocialis, ValueNode)

    for node in nodes_list:
        # Schütze Module, Werte, Input-Knoten und Knoten ohne Label
        if not hasattr(node, 'label') or isinstance(node, protected_classes) or node.label.startswith("Q_"): continue

        history = activation_history_local.get(node.label)
        if not history or len(history) < threshold_epochs: continue # Nicht genug Historie

        # Prüfe, ob *alle* letzten Aktivierungen unter dem Threshold lagen
        recent_history = list(history)[-threshold_epochs:]
        if all(act < activation_threshold for act in recent_history):
            nodes_to_remove_labels.add(node.label)

    if not nodes_to_remove_labels: return nodes_list, 0

    # Filtere Knotenliste
    new_nodes_list = [node for node in nodes_list if node.label not in nodes_to_remove_labels]
    nodes_removed_count = len(nodes_list) - len(new_nodes_list)
    connections_removed_count = 0

    # Entferne eingehende Verbindungen zu gelöschten Knoten
    for node in new_nodes_list:
        if hasattr(node, 'connections'):
            original_conn_count = len(node.connections)
            node.connections = [conn for conn in node.connections if conn.target_node.label not in nodes_to_remove_labels]
            connections_removed_count += original_conn_count - len(node.connections)

    if nodes_removed_count > 0:
         print(f"[Plasticity] Pruned {nodes_removed_count} inactive nodes and {connections_removed_count} related connections.")

    return new_nodes_list, nodes_removed_count


# --- Klassen für Netzwerkstruktur ---
class Connection:
    def __init__(self, target_node, weight=None): self.target_node = target_node; self.weight = weight if weight is not None else random.uniform(0.05, 0.3)
class Node:
    def __init__(self, label: str, neuron_type: str = "excitatory"): self.label = label; self.neuron_type = neuron_type; self.connections = []; self.activation = 0.0; self.activation_sum = 0.0; self.activation_history = deque(maxlen=HISTORY_MAXLEN)
    def add_connection(self, target_node, weight=None):
        if target_node is self or target_node is None: return
        if not any(conn.target_node == target_node for conn in self.connections): self.connections.append(Connection(target_node, weight))
    def __repr__(self): return f"<{type(self).__name__} {self.label} Act:{self.activation:.2f}>"
class MemoryNode(Node):
    def __init__(self, label: str, memory_type="short_term", neuron_type="excitatory"): super().__init__(label, neuron_type=neuron_type); self.memory_type = memory_type; self.retention_times = {"short_term": 5, "mid_term": 20, "long_term": 100}; self.retention_time = self.retention_times.get(memory_type, 20); self.time_in_memory = 0; self.history_length_maps = DEFAULT_HISTORY_LENGTH_MAP_PROMOTION
    def promote(self, activation_threshold=DEFAULT_ACTIVATION_THRESHOLD_PROMOTION, history_length_map=None):
        if history_length_map is None: history_length_map = self.history_length_maps
        required_length = history_length_map.get(self.memory_type)
        if required_length is None or self.memory_type == "long_term" or len(self.activation_history) < required_length:
            # Auch wenn nicht promoted, erhöhe Zeit, außer es ist schon Long-Term
            if self.memory_type != "long_term": self.time_in_memory += 1
            return
        recent_history = list(self.activation_history)[-required_length:]
        if not recent_history:
            if self.memory_type != "long_term": self.time_in_memory += 1
            return
        avg_recent_activation = np.mean(recent_history)
        if avg_recent_activation > activation_threshold:
            original_type = self.memory_type
            if self.memory_type == "short_term": self.memory_type = "mid_term"; self.retention_time = self.retention_times.get("mid_term", 20)
            elif self.memory_type == "mid_term": self.memory_type = "long_term"; self.retention_time = self.retention_times.get("long_term", 100)
            if original_type != self.memory_type: self.time_in_memory = 0 # Reset bei Promotion
            # print(f"[Memory] Node '{self.label}' promoted from {original_type} to {self.memory_type}.")
        else:
            if self.memory_type != "long_term": self.time_in_memory += 1


# --- ValueNode Klasse ---
class ValueNode(Node):
    def __init__(self, label: str, initial_value: float = 0.5): super().__init__(label, neuron_type="excitatory"); self.activation = np.clip(initial_value, 0.0, 1.0)
    def update_value(self, adjustment: float): self.activation = np.clip(self.activation + adjustment, 0.0, 1.0)


# --- Persistentes Gedächtnis Manager ---
class PersistentMemoryManager:
    _instance = None # Singleton Instanz

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(PersistentMemoryManager, cls).__new__(cls)
            # Initialisierung nur beim ersten Erstellen
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, db_path=PERSISTENT_MEMORY_DB):
        if self._initialized: return # Verhindere Re-Initialisierung
        self.db_path = db_path
        self.table_name = PERSISTENT_MEMORY_TABLE
        self.reflection_table_name = PERSISTENT_REFLECTION_TABLE
        self.conn = None
        self.cursor = None
        self._initialize_db()
        self._initialized = True
        print(f"PersistentMemoryManager initialisiert mit DB: {self.db_path}")


    def _get_connection(self):
        """Stellt sicher, dass eine DB-Verbindung besteht."""
        if self.conn is None:
            try:
                # Timeout erhöht, check_same_thread=False für GUI-Thread-Nutzung wichtig
                self.conn = sqlite3.connect(self.db_path, timeout=10, check_same_thread=False)
                self.cursor = self.conn.cursor()
            except sqlite3.Error as e:
                print(f"FEHLER DB Connect ({self.db_path}): {e}")
                self.conn, self.cursor = None, None
                raise # Fehler weitergeben, damit aufrufende Funktion informiert ist
        return self.conn, self.cursor

    def _close_connection(self):
         """Schließt die DB-Verbindung, falls offen."""
         if self.conn:
             try:
                 self.conn.commit() # Änderungen sichern
                 self.conn.close()
             except sqlite3.Error as e:
                 print(f"FEHLER DB Close: {e}")
             finally:
                  self.conn, self.cursor = None, None

    def _execute_query(self, query, params=(), fetch_one=False, fetch_all=False, commit=False):
        """Führt eine SQL-Abfrage sicher aus."""
        result = None
        try:
            conn, cursor = self._get_connection()
            if cursor:
                 cursor.execute(query, params)
                 if commit: conn.commit()
                 if fetch_one: result = cursor.fetchone()
                 elif fetch_all: result = cursor.fetchall()
            # Hinweis: Die Verbindung wird hier NICHT geschlossen, um sie wiederverwenden zu können.
            # Sie sollte explizit geschlossen werden, z.B. am Ende des Programms oder bei Fehlern.
        except sqlite3.Error as e:
            print(f"FEHLER DB Query: {e}\nQuery: {query}\nParams: {params}")
            self._close_connection() # Verbindung bei Fehler schließen
        except Exception as e: # Andere Fehler abfangen
            print(f"FEHLER Ausführung DB Query: {e}")
            self._close_connection()
        return result


    def _initialize_db(self):
        try:
            # Verwende _execute_query für die Initialisierung
            create_memory_table = f'''CREATE TABLE IF NOT EXISTS {self.table_name} (
                                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                                    memory_key TEXT UNIQUE NOT NULL,
                                    memory_content TEXT NOT NULL,
                                    relevance REAL DEFAULT 0.5,
                                    last_accessed REAL DEFAULT 0,
                                    created_at REAL DEFAULT {time.time()}
                                 )'''
            self._execute_query(create_memory_table, commit=True)
            create_memory_index = f'CREATE INDEX IF NOT EXISTS idx_memory_key ON {self.table_name}(memory_key)'
            self._execute_query(create_memory_index, commit=True)

            create_reflection_table = f'''CREATE TABLE IF NOT EXISTS {self.reflection_table_name} (
                                        log_id INTEGER PRIMARY KEY AUTOINCREMENT,
                                        epoch INTEGER,
                                        timestamp REAL,
                                        message TEXT,
                                        log_data TEXT
                                     )'''
            self._execute_query(create_reflection_table, commit=True)
            # Verbindung nach Initialisierung erstmal schließen
            self._close_connection()
            print("Datenbank initialisiert/überprüft.")
        except Exception as e:
            print(f"SCHWERWIEGENDER FEHLER bei DB-Initialisierung: {e}")


    def store_memory(self, key: str, content: dict, relevance: float):
        try:
            content_json = json.dumps(content)
            timestamp = time.time()
            query = f'''INSERT INTO {self.table_name} (memory_key, memory_content, relevance, last_accessed, created_at)
                        VALUES (?, ?, ?, ?, ?)
                        ON CONFLICT(memory_key) DO UPDATE SET
                           memory_content=excluded.memory_content,
                           relevance=excluded.relevance,
                           last_accessed=excluded.last_accessed'''
            self._execute_query(query, (key, content_json, relevance, timestamp, timestamp), commit=True)
        except (TypeError, sqlite3.Error) as e: # Specific error types
            print(f"FEHLER DB Store Memory ('{key}'): {e}")

    def retrieve_memory(self, key: str) -> Optional[dict]:
        try:
            query = f'SELECT memory_content FROM {self.table_name} WHERE memory_key = ?'
            result = self._execute_query(query, (key,), fetch_one=True)
            if result and result[0]:
                # Update last_accessed nur bei erfolgreichem Abruf
                update_query = f'UPDATE {self.table_name} SET last_accessed = ? WHERE memory_key = ?'
                self._execute_query(update_query, (time.time(), key), commit=True)
                try:
                    return json.loads(result[0])
                except json.JSONDecodeError:
                    print(f"FEHLER: Ungültiges JSON in DB für Key '{key}'")
                    return None
            return None
        except Exception as e: # Catch other potential errors during retrieval
            print(f"FEHLER DB Retrieve Memory ('{key}'): {e}")
            return None


    def retrieve_relevant_memories(self, min_relevance: float = MEMORY_RELEVANCE_THRESHOLD, limit: int = 5) -> list:
        memories = []
        try:
            query = f'''SELECT memory_key, memory_content FROM {self.table_name}
                        WHERE relevance >= ? ORDER BY relevance DESC, last_accessed DESC LIMIT ?'''
            results = self._execute_query(query, (min_relevance, limit), fetch_all=True)
            if results:
                 for row in results:
                     try: memories.append({"key": row[0], "content": json.loads(row[1])})
                     except json.JSONDecodeError: print(f"Warnung: Ungültiges JSON in DB für Key {row[0]} bei Relevant Retrieval")
            return memories
        except Exception as e:
             print(f"FEHLER DB Retrieve Relevant: {e}")
             return []


    def store_reflection(self, entry: dict):
         try:
             log_data_json = json.dumps(entry.get('data', {}))
             query = f'''INSERT INTO {self.reflection_table_name} (epoch, timestamp, message, log_data)
                         VALUES (?, ?, ?, ?)'''
             self._execute_query(query, (entry.get('epoch'), entry.get('timestamp', time.time()), entry.get('message'), log_data_json), commit=True)
         except (TypeError, sqlite3.Error) as e:
             print(f"FEHLER DB Store Reflection: {e}")


    def retrieve_reflections(self, limit=20) -> list:
         logs = []
         try:
             query = f'SELECT epoch, timestamp, message, log_data FROM {self.reflection_table_name} ORDER BY timestamp DESC LIMIT ?'
             results = self._execute_query(query, (limit,), fetch_all=True)
             if results:
                 for row in results:
                     try: data = json.loads(row[3] or '{}') # Default to empty dict if None/empty
                     except json.JSONDecodeError: data = {"error": "Invalid JSON in DB"}
                     logs.append({"epoch": row[0], "timestamp": row[1], "message": row[2], "data": data})
             return logs
         except Exception as e:
              print(f"FEHLER DB Retrieve Reflections: {e}")
              return []

    def close(self):
         """Schließt die Datenbankverbindung explizit."""
         print("Schließe Datenbankverbindung...")
         self._close_connection()


# --- Spezialisierte Modul-Klassen ---

class LimbusAffektus(Node):
    """Verwaltet Emotionen (PAD)."""
    def __init__(self, label: str = "Limbus Affektus", neuron_type: str = "interneuron"):
        super().__init__(label, neuron_type=neuron_type)
        self.emotion_state = INITIAL_EMOTION_STATE.copy()

    def update_emotion_state(self, all_nodes: list, module_outputs: dict) -> dict:
        """Aktualisiert den Emotionszustand basierend auf Netzwerkaktivität und Kritik."""
        global CURRENT_EMOTION_STATE, activation_history # Zugriff auf globale Aktivierungshistorie

        pleasure_signal = 0.0
        pos_triggers = ["chance", "wachstum", "positiv", "zufrieden", "erfolg", "gut", "hoch", "ja", "innovation", "lösung"]
        neg_triggers = ["risiko", "problem", "negativ", "unzufrieden", "fehler", "schlecht", "gering", "nein", "kritik", "sicherheit", "bedrohung"]

        relevant_nodes = [n for n in all_nodes if hasattr(n, 'activation') and n.activation > 0.15]

        # Semantische Analyse der Labels
        for node in relevant_nodes:
            if not hasattr(node, 'label'): continue
            label_lower = node.label.lower()
            is_pos = any(trigger in label_lower for trigger in pos_triggers)
            is_neg = any(trigger in label_lower for trigger in neg_triggers)
            if is_pos and not is_neg: pleasure_signal += node.activation * 0.8
            elif is_neg and not is_pos: pleasure_signal -= node.activation * 1.0

        # Einfluss durch Kritiker-Modul
        critic_evals_deque = module_outputs.get("Cortex Criticus")
        if critic_evals_deque and isinstance(critic_evals_deque[-1], list):
            last_eval = critic_evals_deque[-1]
            if last_eval:
                 scores = [e.get('score', 0.5) for e in last_eval if isinstance(e, dict)]
                 if scores:
                    avg_score = np.mean(scores)
                    pleasure_signal += (avg_score - 0.5) * 2.5

        # Arousal: Durchschnitt, Streuung, Änderung
        activations = [n.activation for n in relevant_nodes if not np.isnan(n.activation)]
        avg_act = np.mean(activations) if activations else 0.0
        std_act = np.std(activations) if len(activations) > 1 else 0.0
        # Änderung berechnen anhand der globalen Aktivierungshistorie
        # Sicherstellen, dass activation_history existiert und nicht leer ist
        last_avg_activation = 0.0
        if activation_history:
            all_last_acts = [h[-1] for h in activation_history.values() if h]
            if all_last_acts:
                 last_avg_activation = np.mean(all_last_acts)

        change = abs(avg_act - last_avg_activation) if activation_history else 0.0
        arousal_signal = np.clip(avg_act * 0.5 + std_act * 0.3 + change * 5.0, 0, 1)

        # Dominance: Meta-Kognition und Stabilität
        meta_cog_act = next((n.activation for n in all_nodes if isinstance(n, MetaCognitio)), 0.0)
        control_proxy = 1.0 - std_act
        dominance_signal = np.clip(meta_cog_act * 0.6 + control_proxy * 0.4, 0, 1)

        # --- Emotionszustand aktualisieren ---
        for dim, signal in [("pleasure", pleasure_signal), ("arousal", arousal_signal), ("dominance", dominance_signal)]:
            current_val = self.emotion_state.get(dim, 0.0)
            target_val = np.clip(signal, -1.0, 1.0) if dim == "pleasure" else np.clip(signal * 2.0 - 1.0, -1.0, 1.0)
            decayed_val = current_val * (1 - EMOTION_DECAY_TO_NEUTRAL)
            change_emo = (target_val - decayed_val) * EMOTION_UPDATE_RATE
            noise = np.random.normal(0, EMOTION_VOLATILITY)
            self.emotion_state[dim] = np.clip(decayed_val + change_emo + noise, -1.0, 1.0)

        CURRENT_EMOTION_STATE = self.emotion_state.copy()
        self.activation = np.mean(np.abs(list(self.emotion_state.values())))
        self.activation_history.append(self.activation)
        return self.emotion_state

    def get_emotion_influence_factors(self) -> dict:
        """Liefert emotionale Modulatoren für Netzwerkprozesse."""
        p = self.emotion_state.get("pleasure", 0.0)
        a = self.emotion_state.get("arousal", 0.0)
        d = self.emotion_state.get("dominance", 0.0)
        return {
            "signal_modulation": 1.0 + p * 0.20,
            "learning_rate_factor": np.clip(1.0 + a * 0.3 + p * 0.15, 0.5, 2.0),
            "exploration_factor": np.clip(1.0 + a * 0.4 - d * 0.25, 0.5, 1.8),
            "criticism_weight_factor": np.clip(1.0 - p * 0.3 + d * 0.15, 0.5, 1.5),
            "creativity_weight_factor": np.clip(1.0 + p * 0.25 + a * 0.15, 0.5, 1.8),
        }

class MetaCognitio(Node):
    def __init__(self, label="Meta Cognitio", neuron_type="interneuron"): super().__init__(label, neuron_type=neuron_type); self.reflection_log = deque(maxlen=REFLECTION_LOG_MAX_LEN); self.strategy_state = {"lr_boost": 1.0, "last_avg_activation": 0.5, "stagnation_counter": 0, "oscillation_detected": False}
    def log_reflection(self, message: str, epoch: int, data: Optional[dict] = None):
        log_entry = {"epoch": epoch, "timestamp": time.time(), "message": message, "data": data or {}}
        self.reflection_log.append(log_entry)
        # Speichere Reflexion persistent, wenn Manager verfügbar
        if persistent_memory_manager:
             persistent_memory_manager.store_reflection(log_entry)

    def analyze_network_state(self, all_nodes: list, activation_history_local: dict, weights_history_local: dict, epoch: int):
        nodes_with_history = [n for n in all_nodes if hasattr(n, 'activation') and n.label in activation_history_local and activation_history_local[n.label]]
        if not nodes_with_history: return

        activations = [n.activation for n in nodes_with_history]
        avg_activation = np.mean(activations) if activations else 0.0
        std_activation = np.std(activations) if len(activations) > 1 else 0.0
        activation_change = abs(avg_activation - self.strategy_state["last_avg_activation"])

        # Stagnationserkennung
        if activation_change < STAGNATION_THRESHOLD and avg_activation < 0.65: self.strategy_state["stagnation_counter"] += 1
        else:
            if self.strategy_state["stagnation_counter"] >= STAGNATION_DETECTION_WINDOW:
                self.log_reflection(f"Stagnation überwunden (AvgAct {avg_activation:.3f})", epoch)
                self.adapt_strategy("stagnation_resolved")
            self.strategy_state["stagnation_counter"] = 0
        if self.strategy_state["stagnation_counter"] == STAGNATION_DETECTION_WINDOW:
            self.log_reflection(f"Stagnation vermutet (AvgAct {avg_activation:.3f}, Change {activation_change:.4f})", epoch)
            self.adapt_strategy("stagnation")
        self.strategy_state["last_avg_activation"] = avg_activation

        # Oszillationserkennung
        oscillating_nodes = []
        for label, history in activation_history_local.items():
            if len(history) >= OSCILLATION_DETECTION_WINDOW:
                std_dev = np.std(list(history)[-OSCILLATION_DETECTION_WINDOW:])
                if std_dev > OSCILLATION_THRESHOLD_STD: oscillating_nodes.append(label)
        if oscillating_nodes and not self.strategy_state["oscillation_detected"]:
             self.log_reflection(f"Oszillationen detektiert in: {oscillating_nodes[:3]}...", epoch, data={"nodes": oscillating_nodes})
             self.adapt_strategy("oscillation"); self.strategy_state["oscillation_detected"] = True
        elif not oscillating_nodes and self.strategy_state["oscillation_detected"]:
             self.log_reflection("Oszillationen scheinen reduziert.", epoch)
             self.adapt_strategy("oscillation_resolved"); self.strategy_state["oscillation_detected"] = False

        # Erfolgskontrolle: Beförderte Knoten
        promoted_nodes_this_epoch = [n.label for n in all_nodes if isinstance(n, MemoryNode) and n.time_in_memory == 0 and n.memory_type != "short_term"]
        if promoted_nodes_this_epoch: self.log_reflection(f"Netzwerk-Lernen: Knoten befördert: {promoted_nodes_this_epoch}", epoch)

        self.activation = np.clip(1.0 - std_activation, 0.1, 0.9)
        self.activation_history.append(self.activation)

    def adapt_strategy(self, condition: str):
        """Passt interne Strategien an, z.B. Lernraten-Boost."""
        lr_boost_before = self.strategy_state["lr_boost"]
        if condition == "stagnation": self.strategy_state["lr_boost"] = min(lr_boost_before * 1.2, 2.5)
        elif condition == "oscillation": self.strategy_state["lr_boost"] = max(lr_boost_before * 0.8, 0.5)
        elif condition in ["stagnation_resolved", "oscillation_resolved"]: self.strategy_state["lr_boost"] = lr_boost_before * 0.95 + 1.0 * 0.05 # Langsam zurück zu 1.0
        else: self.strategy_state["lr_boost"] = 1.0 # Reset
        # Logge nur signifikante Änderungen
        if abs(self.strategy_state["lr_boost"] - lr_boost_before) > 0.01:
             print(f"[Meta] Strategieanpassung: {condition} -> LR Boost auf {self.strategy_state['lr_boost']:.2f}")

    def get_meta_cognitive_state(self) -> dict: return self.strategy_state.copy()

class CortexCreativus(Node):
    def __init__(self, label: str = "Cortex Creativus", neuron_type: str | None = None):
        super().__init__(label, neuron_type or random_neuron_type())

    def generate_new_ideas(self, active_nodes: list[Node], creativity_factor: float = 1.0) -> list[str]:
        """Erzeugt neue Ideen basierend auf aktiven Knoten und Kreativitätsfaktor."""
        ideas = []
        if not active_nodes: return ideas
        threshold = max(0.2, 0.6 / max(creativity_factor, 0.1))
        relevant_nodes = [n for n in active_nodes if hasattr(n, 'activation') and n.activation > threshold]
        relevant_nodes.sort(key=lambda n: n.activation, reverse=True)
        num_ideas_to_generate = int(1 + creativity_factor * 1.5)

        # Kombiniere Top-Knoten
        if len(relevant_nodes) >= 2:
            for i in range(min(num_ideas_to_generate, len(relevant_nodes) -1)):
                 node1, node2 = relevant_nodes[i], relevant_nodes[i+1]
                 ideas.append(f"Idea_combining_{node1.label}_and_{node2.label}_(Act:{node1.activation:.2f}+{node2.activation:.2f})")
        elif len(relevant_nodes) == 1: ideas.append(f"Idea_inspired_by_{relevant_nodes[0].label}_(Act:{relevant_nodes[0].activation:.2f})")

        # Wilde Ideen
        if creativity_factor > 1.2 or (len(ideas) < num_ideas_to_generate and active_nodes):
             try:
                 random_node1 = random.choice(active_nodes)
                 potential_partners = [n for n in active_nodes if n != random_node1] # Korrekter Name
                 wild_idea = f"Wild_idea_focusing_on_{random_node1.label}" # Default
                 # Verwende den korrekten Variablennamen 'potential_partners'
                 if potential_partners:
                     # Wähle einen Partner aus potential_partners
                     partner_node = random.choice(potential_partners)
                     # Kürze die Labels sicher, falls sie kurz sind
                     label1_short = random_node1.label[:10] if hasattr(random_node1, 'label') else 'Node1'
                     label2_short = partner_node.label[:10] if hasattr(partner_node, 'label') else 'Node2'
                     wild_idea = f"Wild_link_{label1_short}_{label2_short}"

                 if wild_idea not in ideas: ideas.append(wild_idea)
             except IndexError: # Korrekt eingerückt
                 pass # Ignoriere Fehler, wenn Listen leer sind
        return ideas[:num_ideas_to_generate]

class SimulatrixNeuralis(Node):
    def __init__(self, label: str = "Simulatrix Neuralis", neuron_type: str | None = None):
        super().__init__(label, neuron_type or random_neuron_type())

    def simulate_scenarios(self, active_nodes: list[Node]) -> list[str]:
        """Simuliert Szenarien basierend auf aktiven Knoten und Emotionen."""
        scenarios = []
        if not active_nodes: return scenarios
        pleasure = CURRENT_EMOTION_STATE.get("pleasure", 0.0)
        mood_modifier = "Optimistic" if pleasure > 0.3 else ("Pessimistic" if pleasure < -0.3 else "Neutral")
        scenario_nodes = [n for n in active_nodes if hasattr(n, 'activation') and n.activation > 0.65]
        scenario_nodes.sort(key=lambda n: n.activation, reverse=True)

        for node in scenario_nodes[:3]: # Top 3
            scenarios.append(f"{mood_modifier}Scenario_if_{node.label}_dominates_(Act:{node.activation:.2f})")
            # Optional: Werte-basierte Varianten
            value_nodes_dict = {v.label: v.activation for v in active_nodes if isinstance(v, ValueNode)}
            if value_nodes_dict.get("Sicherheit", 0) > 0.6 and "risiko" not in node.label.lower(): scenarios.append(f"CautiousVariant_of_{node.label}")
            if value_nodes_dict.get("Innovation", 0) > 0.6 and "chance" not in node.label.lower(): scenarios.append(f"InnovativeVariant_of_{node.label}")
        return scenarios

class CortexCriticus(Node):
    def __init__(self, label: str = "Cortex Criticus", neuron_type: str | None = None):
        super().__init__(label, neuron_type or "inhibitory")

    def evaluate_ideas(self, items_to_evaluate: list[str], current_network_state_nodes: list[Node], criticism_factor: float = 1.0) -> list[dict]:
        """Bewertet Ideen/Szenarien kritisch."""
        evaluated = []
        if not items_to_evaluate: return evaluated
        value_nodes = {n.label: n.activation for n in current_network_state_nodes if isinstance(n, ValueNode)}
        sicherheit_val = value_nodes.get("Sicherheit", 0.5)
        ethik_val = value_nodes.get("Ethik", 0.5)
        pleasure = CURRENT_EMOTION_STATE.get("pleasure", 0.0)
        base_criticism = 0.5 + (criticism_factor - 1.0) * 0.2 - pleasure * 0.15

        for item in items_to_evaluate:
            score = 0.5; adjustment = 0.0
            item_lower = item.lower()
            if "risiko" in item_lower or "problem" in item_lower or "pessimistic" in item_lower: adjustment -= 0.2 * sicherheit_val
            if "chance" in item_lower or "neu" in item_lower or "optimistic" in item_lower: adjustment += 0.1 * (1.0 - sicherheit_val)
            if "ethik" in item_lower or "moral" in item_lower: adjustment += 0.15 * ethik_val
            if "wild_idea" in item_lower: adjustment -= 0.1 * criticism_factor
            if "cautiousvariant" in item_lower: adjustment += 0.05 * sicherheit_val # Vorsichtige Varianten leicht positiv, wenn Sicherheit hoch
            if "innovativevariant" in item_lower: adjustment += 0.05 * (1 - sicherheit_val) # Innovative positiv, wenn Sicherheit niedrig

            raw_score = base_criticism + adjustment + random.uniform(-0.05, 0.05) # Weniger Rauschen
            score = np.clip(raw_score, 0.0, 1.0)
            evaluated.append({"item": item, "score": round(score, 3)}) # Key geändert zu "item"
        return evaluated

class CortexSocialis(Node):
    def __init__(self, label: str = "Cortex Socialis", neuron_type: str | None = None):
        super().__init__(label, neuron_type or random_neuron_type())

    def update_social_factors(self, social_network: dict[str, float], active_nodes: list[Node]) -> dict[str, float]:
        """Aktualisiert soziale Faktoren basierend auf Knotenaktivierung und Dominanz."""
        dominance = CURRENT_EMOTION_STATE.get("dominance", 0.0)
        global_influence_mod = 1.0 + dominance * 0.1
        updated_social_network = social_network.copy()

        # Nur Kategorie-Knoten für soziale Faktoren berücksichtigen?
        target_nodes = [n for n in active_nodes if isinstance(n, MemoryNode)]

        for node in target_nodes:
            label = getattr(node, 'label', None)
            activation = getattr(node, 'activation', 0.0)
            if label is None or label not in updated_social_network: continue

            current_factor = updated_social_network[label]
            change_factor = 0.0
            if activation > 0.75: change_factor = 0.05
            elif activation < 0.25: change_factor = -0.03
            new_factor = current_factor + change_factor * global_influence_mod
            updated_social_network[label] = np.clip(new_factor, 0.05, 0.95)
        return updated_social_network


# --- Netzwerk-Initialisierung & Verbindung  ---
def initialize_network_nodes(
    categories: list[str] | np.ndarray | pd.Series,
    initial_values: dict[str, float] = DEFAULT_VALUES
) -> tuple[list[MemoryNode], list[Node], list[ValueNode]]:
    """Initialisiert Knoten für Kategorien, Module und Werte."""
    global CURRENT_EMOTION_STATE
    CURRENT_EMOTION_STATE = INITIAL_EMOTION_STATE.copy()

    if not isinstance(categories, (list, np.ndarray, pd.Series)) or len(categories) == 0:
        print("FEHLER: Ungültige Kategorienliste für Initialisierung.")
        return [], [], []

    # Eindeutige Kategorien sicherstellen und MemoryNodes erstellen
    if isinstance(categories, (np.ndarray, list)):
        categories_series = pd.Series(categories)
    elif isinstance(categories, pd.Series):
        categories_series = categories.copy() # Arbeite mit Kopie
    else:
         print(f"WARNUNG: Unerwarteter Typ für Kategorien: {type(categories)}. Versuche Konvertierung.")
         try:
             categories_series = pd.Series(list(categories))
         except Exception as e:
             print(f"FEHLER: Konnte Kategorien nicht in Series umwandeln: {e}")
             return [], [], []

    # Bereinige die Series und erhalte eindeutige Werte
    cleaned_categories = categories_series.astype(str).str.strip().replace('', 'Unkategorisiert')
    # pd.Series.unique() gibt direkt ein Numpy-Array zurück, das für die Sortierung geeignet ist
    unique_values = cleaned_categories.unique()
    # Konvertiere zu Liste und sortiere
    unique_categories = sorted(list(unique_values))

    if not unique_categories or (len(unique_categories) == 1 and unique_categories[0] == 'Unkategorisiert'):
        # Optional: Behandle den Fall, dass nach der Bereinigung keine sinnvollen Kategorien übrig bleiben
        print("WARNUNG: Keine gültigen Kategorien nach Bereinigung gefunden oder nur 'Unkategorisiert'.")
        # Eventuell hier return [], [], [] wenn keine Kategorien sinnvoll sind? Hängt von der gewünschten Logik ab.
        # Aktuell wird 'Unkategorisiert' als gültige Kategorie behandelt, wenn sie vorkommt.

    print(f"Initialisiere Netzwerk: {len(unique_categories)} Kategorien, {len(initial_values)} Werte...")
    
    
    
    
    
    
    if not unique_categories:
        print("FEHLER: Keine gültigen Kategorien nach Bereinigung.")
        return [], [], []
    print(f"Initialisiere Netzwerk: {len(unique_categories)} Kategorien, {len(initial_values)} Werte...")
    category_nodes = [MemoryNode(label=cat) for cat in unique_categories]

    # Module initialisieren
    module_nodes = [
        CortexCreativus(), SimulatrixNeuralis(), CortexCriticus(),
        LimbusAffektus(), MetaCognitio(), CortexSocialis()
    ]

    # Werte-Knoten initialisieren
    value_nodes = [ValueNode(label=key, initial_value=value) for key, value in initial_values.items()]

    print(f"Init abgeschlossen: {len(category_nodes)} Kategorien, {len(module_nodes)} Module, {len(value_nodes)} Werte.")
    return category_nodes, module_nodes, value_nodes

def connect_network_components(
    category_nodes: list[MemoryNode],
    module_nodes: list[Node],
    question_nodes: list[Node],
    value_nodes: list[ValueNode]
) -> list[Node]:
    """Verbindet alle Knotenarten untereinander nach definierten Regeln."""
    print("Verbinde Netzwerkkomponenten...")
    all_nodes = category_nodes + module_nodes + question_nodes + value_nodes
    node_map = {node.label: node for node in all_nodes if hasattr(node, 'label')}

    connection_specs = [
        {"source_list": module_nodes, "target_list": module_nodes, "prob": 0.4, "weight_range": (0.05, 0.15), "bidirectional": True},
        {"source_list": module_nodes, "target_list": category_nodes, "prob": 0.6, "weight_range": (0.1, 0.25), "bidirectional": False},
        {"source_list": category_nodes, "target_list": module_nodes, "prob": 0.5, "weight_range": (0.05, 0.15), "bidirectional": False},
        {"source_list": category_nodes, "target_list": category_nodes, "prob": 0.08, "weight_range": (0.02, 0.1), "bidirectional": True},
        {"source_list": question_nodes, "target_list": category_nodes, "prob": 1.0, "weight_range": (0.8, 1.0), "bidirectional": False, "special": "question_to_category"},
        {"source_list": value_nodes, "target_list": module_nodes, "prob": 1.0, "weight_range": (0.1, 0.4), "bidirectional": False, "special": "value_to_module"},
        {"source_list": module_nodes, "target_list": value_nodes, "prob": 1.0, "weight_range": (0.05, 0.2), "bidirectional": False, "special": "module_to_value"},
        {"source_list": value_nodes, "target_list": category_nodes, "prob": 0.2, "weight_range": (0.05, 0.15), "bidirectional": True, "special": "value_category_thematic"},
    ]

    connections_created = 0
    for spec in connection_specs:
        source_list, target_list = spec["source_list"], spec["target_list"]
        prob, (w_min, w_max), bidi = spec["prob"], spec["weight_range"], spec["bidirectional"]
        special = spec.get("special")

        for i, src in enumerate(source_list):
            start_j = i + 1 if (source_list is target_list and bidi) else 0
            for j in range(start_j, len(target_list)):
                tgt = target_list[j]
                if src == tgt: continue
                if random.random() >= prob and special is None: continue

                weight = random.uniform(w_min, w_max)
                connected = False

                if special == "question_to_category":
                    try:
                        q_label_parts = src.label.split('_')
                        # Erwartetes Format: Q_idx_Category_OptionalText...
                        if len(q_label_parts) >= 3:
                             cat_label_from_q = q_label_parts[2]
                             # Finde den passenden Kategorie-Knoten (Groß/Kleinschreibung ignorieren?)
                             if tgt.label.lower() == cat_label_from_q.lower(): # Toleranter Vergleich
                                src.add_connection(tgt, weight=weight)
                                connections_created += 1; connected = True
                    except (IndexError, AttributeError): continue
                elif special == "value_to_module":
                    v_m_map = {"Innovation": ("Cortex Creativus", 1.0), "Sicherheit": ("Cortex Criticus", 1.0), "Effizienz": ("Meta Cognitio", 0.8), "Neugier": ("Cortex Creativus", 0.5), "Ethik": ("Cortex Criticus", 0.9)}
                    if src.label in v_m_map:
                        tgt_mod_label, factor = v_m_map[src.label]
                        if tgt.label == tgt_mod_label: src.add_connection(tgt, weight * factor); connections_created += 1; connected = True
                elif special == "module_to_value":
                     m_v_map = {"Cortex Creativus": ("Innovation", 0.7), "Cortex Criticus": ("Sicherheit", 0.7), "Meta Cognitio": ("Effizienz", 0.6), "CortexCriticus": ("Ethik", 0.5)} # Criticus beeinflusst Sicherheit UND Ethik
                     # Korrigiere Map-Key für Ethik
                     if src.label == "Cortex Criticus" and tgt.label == "Ethik":
                           factor = m_v_map["CortexCriticus"][1] # Nimm Faktor für Ethik
                           src.add_connection(tgt, weight * factor); connections_created += 1; connected = True
                     elif src.label in m_v_map and m_v_map[src.label][0] == tgt.label: # Für andere M->V
                         tgt_val_label, factor = m_v_map[src.label]
                         src.add_connection(tgt, weight * factor); connections_created += 1; connected = True
                elif special == "value_category_thematic":
                    cat_lower = tgt.label.lower(); connect = False
                    if src.label == "Sicherheit" and ("risiko" in cat_lower or "problem" in cat_lower): connect = True
                    if src.label == "Innovation" and ("chance" in cat_lower or "neu" in cat_lower): connect = True
                    if src.label == "Ethik" and "ethik" in cat_lower: connect = True
                    if connect:
                        src.add_connection(tgt, weight * 1.5)
                        if bidi: tgt.add_connection(src, weight * 0.5)
                        connections_created += (2 if bidi else 1); connected = True
                elif special is None: # Standardverbindung
                    src.add_connection(tgt, weight); connections_created += 1
                    if bidi:
                         tgt.add_connection(src, random.uniform(w_min * 0.8, w_max * 0.8)); connections_created += 1
                    connected = True

    print(f"Netzwerkkomponenten verbunden ({connections_created} Verbindungen erstellt).")
    return all_nodes


# --- Signalpropagation ---
def propagate_signal(node, current_activation, emotion_factors, value_state):
    """Propagiert das Signal eines Knotens an seine verbundenen Ziele."""
    if not hasattr(node, 'connections') or not node.connections or current_activation < 0.01: return

    base_signal = current_activation
    signal_modulation = emotion_factors.get("signal_modulation", 1.0)
    modulated_signal = base_signal * signal_modulation

    for connection in node.connections:
        signal_strength = modulated_signal * connection.weight
        if node.neuron_type == "inhibitory": signal_strength *= -1.5

        target_node = connection.target_node
        if hasattr(target_node, 'activation_sum'):
            target_value_mod = 1.0
            if isinstance(target_node, CortexCriticus): target_value_mod *= 1.0 + value_state.get("Sicherheit", 0.5) * 0.5
            elif isinstance(target_node, CortexCreativus): target_value_mod *= 1.0 + value_state.get("Innovation", 0.5) * 0.5
            elif isinstance(target_node, MetaCognitio): target_value_mod *= 1.0 + value_state.get("Effizienz", 0.5) * 0.4
            final_signal = signal_strength * target_value_mod
            target_node.activation_sum = np.clip(target_node.activation_sum + final_signal, -20, 20)


# --- Kern-Simulationsschleife ---
def simulate_learning_cycle(
    data: pd.DataFrame,
    category_nodes: list,
    module_nodes: list,
    value_nodes: list,
    epochs: int = DEFAULT_EPOCHS,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    reward_interval: int = DEFAULT_REWARD_INTERVAL,
    decay_rate: float = DEFAULT_DECAY_RATE,
    initial_emotion_state: dict = None,
    context_factors: dict | None = None,
    persistent_memory: PersistentMemoryManager = None,
    load_state_from: Optional[str] = None,
    status_callback: Callable[[str], None] = _default_status_callback
):
    global CURRENT_EMOTION_STATE, activation_history # Globale Variablen
    status_callback("Beginne Simulationszyklus...")

    # --- Initialisierung ---
    if initial_emotion_state is None: initial_emotion_state = INITIAL_EMOTION_STATE.copy()
    CURRENT_EMOTION_STATE = initial_emotion_state.copy()
    if context_factors is None: context_factors = {}

    module_dict = {m.label: m for m in module_nodes if hasattr(m, 'label')}
    value_dict  = {v.label: v for v in value_nodes if hasattr(v, 'label')}
    weights_history: Dict[str, deque] = {}
    activation_history.clear() # Leere globale History zu Beginn
    module_outputs_log = {label: deque(maxlen=HISTORY_MAXLEN // 3) for label in module_dict.keys()}
    interpretation_log = []
    value_history: Dict[str, deque] = {v.label: deque(maxlen=HISTORY_MAXLEN) for v in value_nodes}
    question_nodes = []
    cat_node_map = {node.label: node for node in category_nodes if hasattr(node, 'label')}

    # Frage-Knoten erstellen
    for idx, row in data.iterrows():
        cat_label = str(row.get('Kategorie', 'Unkategorisiert')).strip()
        if not cat_label: cat_label = 'Unkategorisiert'
        q_label = f"Q_{idx}_{cat_label}_{str(row.get('Frage', 'Frage?'))[:20]}"
        q_node = Node(q_label, neuron_type="excitatory")
        question_nodes.append(q_node)

    # Netzwerk verbinden
    all_nodes_sim = connect_network_components(category_nodes, module_nodes, question_nodes, value_nodes)

    # Historien initialisieren (jetzt mit allen Knoten)
    activation_history = {node.label: deque(maxlen=HISTORY_MAXLEN) for node in all_nodes_sim if hasattr(node, 'label')}
    for node in all_nodes_sim:
        if hasattr(node, 'connections'):
            for conn in node.connections:
                if hasattr(node, 'label') and hasattr(conn.target_node, 'label'):
                    history_key = f"{node.label} → {conn.target_node.label}"
                    weights_history.setdefault(history_key, deque(maxlen=HISTORY_MAXLEN))

    # --- Laden des Zustands (optional) ---
    if load_state_from and os.path.exists(load_state_from):
        # TODO: Implementiere Lade-Logik
        status_callback(f"Laden von Zustand aus {load_state_from}... (nicht implementiert)")
        pass

    social_network = {node.label: random.uniform(0.2, 0.6) for node in category_nodes if hasattr(node, 'label')}
    base_lr = learning_rate
    current_dr = decay_rate

    status_callback("Starte Epochen-Simulation...")
    limbus_module = module_dict.get("Limbus Affektus")
    meta_cog = module_dict.get("Meta Cognitio")

    # --- Haupt-Simulationsschleife ---
    iterator = range(epochs)
    if TQDM_AVAILABLE:
        iterator = tqdm(iterator, desc="Simulating Cognitive Cycle", unit="epoch")

    for epoch in iterator:
        epoch_start_time = time.time()

        # A. Reset
        for node in all_nodes_sim: node.activation_sum = 0.0

        # B. Kontext holen
        emotion_factors = limbus_module.get_emotion_influence_factors() if isinstance(limbus_module, LimbusAffektus) else {}
        current_value_state = {v.label: v.activation for v in value_nodes}

        # C. Input-Verarbeitung
        for idx, q_node in enumerate(question_nodes):
            if idx < len(data):
                 norm_answer = data['normalized_answer'].iloc[idx]
                 q_node.activation = norm_answer
                 if q_node.label in activation_history: activation_history[q_node.label].append(q_node.activation)
                 propagate_signal(q_node, q_node.activation, emotion_factors, current_value_state)
            else: # Fallback, falls weniger Daten als Fragenknoten (sollte nicht passieren)
                 q_node.activation = 0.0
                 if q_node.label in activation_history: activation_history[q_node.label].append(0.0)

        # D. Signalpropagation
        nodes_to_propagate = category_nodes + module_nodes + value_nodes
        random.shuffle(nodes_to_propagate)
        for node in nodes_to_propagate:
            if node.activation > 0: propagate_signal(node, node.activation, emotion_factors, current_value_state)

        # E. Aktivierungs-Update und History
        all_active_nodes_this_epoch = []
        for node in all_nodes_sim:
            if node not in question_nodes:
                new_activation = sigmoid(node.activation_sum)
                noise_factor = 0.03 * emotion_factors.get("exploration_factor", 1.0)
                node.activation = add_activation_noise(new_activation, noise_level=noise_factor)
            if hasattr(node, 'label') and node.label in activation_history:
                current_act = float(node.activation) if not np.isnan(node.activation) else 0.0
                activation_history[node.label].append(current_act)
            if isinstance(node, ValueNode) and hasattr(node, 'label') and node.label in value_history:
                value_history[node.label].append(float(node.activation))
            if hasattr(node, 'activation') and node.activation > 0.15: all_active_nodes_this_epoch.append(node)
            if isinstance(node, MemoryNode): node.promote()

        # F. Emotions-Update
        if isinstance(limbus_module, LimbusAffektus):
            new_emotion_state = limbus_module.update_emotion_state(all_nodes_sim, module_outputs_log)
            module_outputs_log["Limbus Affektus"].append(new_emotion_state)
            emotion_factors = limbus_module.get_emotion_influence_factors() # Update Faktoren

        # G. Modul-Aktivitäten
        ideas, evaluated, scenarios = [], [], []
        creativity_factor = emotion_factors.get('creativity_weight_factor', 1.0)
        criticism_factor = emotion_factors.get('criticism_weight_factor', 1.0)
        if "Cortex Creativus" in module_dict:
            ideas = module_dict["Cortex Creativus"].generate_new_ideas(all_active_nodes_this_epoch, creativity_factor)
            if ideas: module_outputs_log["Cortex Creativus"].append(ideas)
        if "Simulatrix Neuralis" in module_dict:
            scenarios = module_dict["Simulatrix Neuralis"].simulate_scenarios(all_active_nodes_this_epoch)
            if scenarios: module_outputs_log["Simulatrix Neuralis"].append(scenarios)
        if "Cortex Criticus" in module_dict:
            items_to_evaluate = ideas + scenarios
            if items_to_evaluate:
                 evaluated = module_dict["Cortex Criticus"].evaluate_ideas(items_to_evaluate, all_nodes_sim, criticism_factor)
                 if evaluated: module_outputs_log["Cortex Criticus"].append(evaluated)
        if "Cortex Socialis" in module_dict:
            # Kategorieknoten für soziale Updates verwenden
            current_category_nodes = [n for n in all_nodes_sim if isinstance(n, MemoryNode)]
            social_network = module_dict["Cortex Socialis"].update_social_factors(social_network, current_category_nodes)
            module_outputs_log["Cortex Socialis"].append({"factors_updated": True})
            social_influence(all_nodes_sim, social_network) # Sozialen Einfluss anwenden

        # H. Werte-Update
        for v_node in value_nodes:
            adjustment = calculate_value_adjustment(v_node.label, all_nodes_sim, module_outputs_log, value_dict)
            v_node.update_value(adjustment)          

        # I. Meta-Kognition und adaptives Lernen
        dynamic_lr = base_lr
        if isinstance(meta_cog, MetaCognitio):
            meta_cog.analyze_network_state(all_nodes_sim, activation_history, weights_history, epoch + 1) # Epoche + 1 übergeben
            meta_cognitive_state = meta_cog.get_meta_cognitive_state()
            dynamic_lr = calculate_dynamic_learning_rate(base_lr, CURRENT_EMOTION_STATE, meta_cognitive_state)
            module_outputs_log["Meta Cognitio"].append({"lr": dynamic_lr, "state": meta_cognitive_state})
        for node in all_nodes_sim: hebbian_learning(node, dynamic_lr) # Hebb'sches Lernen anwenden

        # J. Verstärkung & Zerfall
        if (epoch + 1) % reward_interval == 0: apply_reinforcement(all_nodes_sim, module_outputs_log)
        decay_weights(all_nodes_sim, current_dr)

        # K. Plastizität
        if (epoch + 1) % STRUCTURAL_PLASTICITY_INTERVAL == 0:
            # status_callback(f"E{epoch+1}: Strukturelle Plastizität...") # Kann zu viel Output sein
            pruned_conn = prune_connections(all_nodes_sim)
            sprouted_conn = sprout_connections(all_nodes_sim, activation_history)
            all_nodes_sim, pruned_nodes = prune_inactive_nodes(all_nodes_sim, activation_history, epoch + 1, enabled=NODE_PRUNING_ENABLED)
            if pruned_conn or sprouted_conn or pruned_nodes:
                 status_callback(f"E{epoch+1}: Plastizität (-{pruned_conn}c, +{sprouted_conn}c, -{pruned_nodes}n)")
            if pruned_nodes > 0: # Wichtig: Maps/Listen aktualisieren nach Knoten-Pruning
                 node_map = {node.label: node for node in all_nodes_sim if hasattr(node, 'label')}
                 activation_history = {k: v for k, v in activation_history.items() if k in node_map}
                 # weights_history bereinigen (komplexer, vereinfacht: ungültige Einträge ignorieren beim Plotten)

        # L. Logging der Verbindungsgewichte
        for node in all_nodes_sim:
            if hasattr(node, 'connections') and hasattr(node, 'label'):
                for conn in node.connections:
                    if hasattr(conn.target_node, 'label') and conn.target_node.label in activation_history: # Ziel muss existieren
                         history_key = f"{node.label} → {conn.target_node.label}"
                         current_weight = float(conn.weight) if not np.isnan(conn.weight) else 0.0
                         weights_history.setdefault(history_key, deque(maxlen=HISTORY_MAXLEN)).append(current_weight)

        # M. Gedächtniskonsolidierung
        if persistent_memory and (epoch + 1) % MEMORY_CONSOLIDATION_INTERVAL == 0:
            consolidate_memories(all_nodes_sim, persistent_memory, epoch, status_callback)

        # N. Interpretation der Epoche
        current_final_category_nodes = [n for n in all_nodes_sim if isinstance(n, MemoryNode)]
        current_final_module_nodes = [n for n in all_nodes_sim if isinstance(n, (CortexCreativus, SimulatrixNeuralis, CortexCriticus, LimbusAffektus, MetaCognitio, CortexSocialis))]
        current_final_value_nodes = [n for n in all_nodes_sim if isinstance(n, ValueNode)]
        epoch_interpretation = interpret_epoch_state(epoch, current_final_category_nodes, current_final_module_nodes, module_outputs_log, activation_history, current_final_value_nodes)
        interpretation_log.append(epoch_interpretation)

        # O. Optional: Update TQDM description
        if TQDM_AVAILABLE and isinstance(iterator, tqdm):
             dom_cat = epoch_interpretation.get('dominant_category','?')[:15]
             dom_act = epoch_interpretation.get('dominant_activation',0)
             pleasure = CURRENT_EMOTION_STATE.get('pleasure', 0)
             iterator.set_description(f"Simulating E{epoch+1}/{epochs} | Dom: {dom_cat}({dom_act:.2f}) P:{pleasure:.2f}")


    status_callback("Simulationszyklus abgeschlossen.")

    # --- Finale Ergebnisse sammeln ---
    final_activation_history = {k: list(v) for k, v in activation_history.items() if v}
    final_weights_history = {k: list(v) for k, v in weights_history.items() if v}
    final_value_history = {k: list(v) for k, v in value_history.items() if v}
    final_category_nodes = [n for n in all_nodes_sim if isinstance(n, MemoryNode)]
    final_module_nodes = [n for n in all_nodes_sim if isinstance(n, (CortexCreativus, SimulatrixNeuralis, CortexCriticus, LimbusAffektus, MetaCognitio, CortexSocialis))]
    final_value_nodes = [n for n in all_nodes_sim if isinstance(n, ValueNode)]

    status_callback("Sammle finale Netzwerkzustände...")
    return (final_activation_history, final_weights_history, interpretation_log,
            final_category_nodes, final_module_nodes, final_value_nodes, all_nodes_sim, final_value_history)


# --- Hilfsfunktionen für Simulation Cycle ---
def calculate_value_adjustment(value_label: str, all_nodes: list, module_outputs: dict, value_dict: dict) -> float:
    """Berechnet die Anpassung für einen Wert-Knoten."""
    adjustment = 0.0
    # Extrahiere relevante Aktivierungen für einfachere Verwendung
    module_activations = {m.label: m.activation for m in all_nodes if hasattr(m,'label') and isinstance(m, (CortexCreativus, SimulatrixNeuralis, CortexCriticus, LimbusAffektus, MetaCognitio, CortexSocialis))}
    category_activations = {c.label: c.activation for c in all_nodes if isinstance(c, MemoryNode) and hasattr(c, 'label')} # Sicherstellen, dass ein Label vorhanden ist

    # --- Logik für "Innovation" ---
    if value_label == "Innovation":
        # Einfluss von Kreativmodul
        adjustment += (module_activations.get("Cortex Creativus", 0.5) - 0.5) * 0.6
        # Einfluss von "Chancen"-Kategorien
        chance_acts = [a for l, a in category_activations.items() if "chance" in l.lower() or "potential" in l.lower()]
        avg_chance_act = np.mean(chance_acts) if chance_acts else 0.0
        adjustment += (avg_chance_act - 0.4) * 0.4 # Erhöhung, wenn Chancen > 0.4

    # --- Logik für "Sicherheit" ---
    elif value_label == "Sicherheit":
        # Einfluss von Kritikmodul
        adjustment += (module_activations.get("Cortex Criticus", 0.5) - 0.5) * 0.6
        # Einfluss von "Risiko"-Kategorien
        risiko_acts = [a for l, a in category_activations.items() if "risiko" in l.lower() or "problem" in l.lower() or "bedrohung" in l.lower()]
        avg_risiko_act = np.mean(risiko_acts) if risiko_acts else 0.0
        # Erhöhung, wenn Risiken > 0.4; Stärkere Reaktion auf hohe Risiken
        adjustment += (avg_risiko_act - 0.4) * 0.7 # Faktor erhöht für stärkere Reaktion

    # --- Logik für "Effizienz" ---
    elif value_label == "Effizienz":
        # Einfluss von Meta-Kognitionsmodul (höhere Aktivierung -> mehr Effizienz)
        adjustment += (module_activations.get("Meta Cognitio", 0.5) - 0.5) * 0.5 # Angepasst: Steigt, wenn Meta > 0.5

    # --- Logik für "Ethik" ---
    elif value_label == "Ethik":
        # Einfluss von "Ethik"-Kategorien
        ethik_acts = [a for l, a in category_activations.items() if "ethik" in l.lower()]
        avg_ethik_act = np.mean(ethik_acts) if ethik_acts else 0.0
        # Erhöhung, wenn Ethik-Themen > 0.3; Stärkere Reaktion
        adjustment += (avg_ethik_act - 0.3) * 0.7 # Faktor beibehalten oder leicht erhöht

        # Einfluss von Kritikmodul (starke Kritik KANN Ethik temporär senken, WENN Sicherheit niedrig ist)
        critic_deque = module_outputs.get("Cortex Criticus")
        # Sicherstellen, dass die Deque nicht leer ist und das letzte Element eine Liste ist
        if critic_deque and isinstance(critic_deque[-1], list):
             last_eval = critic_deque[-1]
             if last_eval: # Sicherstellen, dass die Liste der Evaluationen nicht leer ist
                 # Filtere gültige Scores (nur Dictionaries mit 'score'-Key)
                 scores = [e.get('score') for e in last_eval if isinstance(e, dict) and 'score' in e and isinstance(e['score'], (int, float))]
                 # *** KORREKTUR HIER ***
                 if scores: # Nur wenn gültige Scores vorhanden sind
                     # Hole das ValueNode-Objekt für "Sicherheit"
                     sicherheit_node = value_dict.get("Sicherheit")
                     # Prüfe, ob der Knoten gefunden wurde und hole seine Aktivierung.
                     # Setze einen Standardwert (z.B. 0.5), falls der Knoten nicht gefunden wurde
                     sicherheit_activation = sicherheit_node.activation if isinstance(sicherheit_node, ValueNode) else 0.5

                     # Vergleiche jetzt den AktivierungsWERT (float) mit 0.5
                     if np.mean(scores) < 0.4 and sicherheit_activation < 0.5:
                         adjustment -= 0.02 # Kleiner Malus, vielleicht etwas stärker

    # --- Logik für "Neugier" ---
    elif value_label == "Neugier":
        pleasure = CURRENT_EMOTION_STATE.get('pleasure', 0.0)
        arousal = CURRENT_EMOTION_STATE.get('arousal', 0.0)
        # Erhöhte Neugier bei moderatem Arousal und nicht-negativer Stimmung
        if arousal > 0.3 and pleasure > -0.2: # Schwellwerte angepasst
            adjustment += 0.06 # Etwas stärkerer Anstieg
        # Einfluss von Innovationswert (nur wenn Innovation bereits hoch ist)
        # Hole Innovations-Node direkt über value_dict für Konsistenz
        innov_node = value_dict.get("Innovation")
        innov_activation = innov_node.activation if isinstance(innov_node, ValueNode) else 0.5
        if innov_activation > 0.65: # Höherer Schwellwert
             adjustment += (innov_activation - 0.65) * 0.08 # Leicht stärkerer Einfluss

    # Wende die globale Update-Rate an
    # Begrenze die maximale Anpassung pro Schritt, um Überschwingen zu verhindern
    max_adjustment_step = 0.05 # Beispielgrenze
    final_adjustment = np.clip(adjustment * VALUE_UPDATE_RATE, -max_adjustment_step, max_adjustment_step)

    return final_adjustment

# --- Implementierung der apply_reinforcement Funktion ---
def apply_reinforcement(all_nodes: list, module_outputs: dict):
    """Wendet Verstärkungslernen auf Verbindungen an."""
    reward_signal = 0.0
    reinforced_connections = 0
    pleasure = CURRENT_EMOTION_STATE.get('pleasure', 0.0)
    if pleasure > REINFORCEMENT_PLEASURE_THRESHOLD:
        reward_signal += (pleasure - REINFORCEMENT_PLEASURE_THRESHOLD) / (1.0 - REINFORCEMENT_PLEASURE_THRESHOLD) * 0.6

    critic_evals_deque = module_outputs.get("Cortex Criticus")
    if critic_evals_deque and isinstance(critic_evals_deque[-1], list):
        last_eval = critic_evals_deque[-1]
        if last_eval:
            scores = [e.get('score', 0.0) for e in last_eval if isinstance(e, dict)]
            if scores:
                avg_score = np.mean(scores)
                if avg_score > REINFORCEMENT_CRITIC_THRESHOLD:
                    reward_signal += (avg_score - REINFORCEMENT_CRITIC_THRESHOLD) / (1.0 - REINFORCEMENT_CRITIC_THRESHOLD) * 0.4

    if reward_signal > 0.1:
        effective_reinforcement = REINFORCEMENT_FACTOR * reward_signal
        for node in all_nodes:
            if hasattr(node, 'activation') and node.activation > 0.3:
                node_act = node.activation
                if hasattr(node, 'connections'):
                    for conn in node.connections:
                        target_node = conn.target_node
                        if hasattr(target_node, 'activation') and target_node.activation > 0.3:
                            target_act = target_node.activation
                            delta_weight = effective_reinforcement * node_act * target_act
                            conn.weight = np.clip(conn.weight + delta_weight, 0.0, 1.0)
                            reinforced_connections += 1
    # if reinforced_connections > 0: print(f"[Reinforcement] Reward {reward_signal:.3f} -> {reinforced_connections} connections reinforced.")


def consolidate_memories(all_nodes: list, pm_manager: PersistentMemoryManager, epoch: int, status_callback):
    """Konsolidiert hochaktive Langzeit-MemoryNodes in die persistente DB."""
    # status_callback(f"E{epoch+1}: Versuche Gedächtniskonsolidierung...") # Kann zu viel Output sein
    consolidated_count = 0
    if pm_manager is None: return # Nichts tun, wenn kein Manager da

    for node in all_nodes:
        if isinstance(node, MemoryNode) and node.memory_type == "long_term" and node.activation > MEMORY_RELEVANCE_THRESHOLD:
             memory_key = f"category_{node.label}"
             content = {
                 "label": node.label, "activation_at_consolidation": round(node.activation, 4),
                 "type": "long_term_category", "consolidation_epoch": epoch + 1,
                 "neuron_type": node.neuron_type # Zusätzliche Info speichern
             }
             relevance = node.activation
             pm_manager.store_memory(memory_key, content, relevance)
             consolidated_count += 1

    # if consolidated_count > 0: status_callback(f"E{epoch+1}: {consolidated_count} Gedächtniselement(e) persistent gespeichert/aktualisiert.")


# --- Interpretation & Report---
def interpret_epoch_state(epoch: int, category_nodes: list, module_nodes: list, module_outputs: dict, activation_history_local: dict, value_nodes: list) -> dict:
    """Interpretiert den Zustand des Netzwerks am Ende einer Epoche."""
    interpretation = {'epoch': epoch + 1}
    valid_category_nodes = [n for n in category_nodes if hasattr(n, 'activation') and isinstance(getattr(n, 'activation', None), (float, np.floating)) and not np.isnan(n.activation)]
    if valid_category_nodes:
        sorted_cats = sorted(valid_category_nodes, key=lambda n: n.activation, reverse=True)
        interpretation['dominant_category'] = getattr(sorted_cats[0], 'label', 'Unbekannt')
        interpretation['dominant_activation'] = round(sorted_cats[0].activation, 4)
        interpretation['category_ranking'] = [(getattr(n, 'label', 'Unbekannt'), round(n.activation, 4)) for n in sorted_cats[:5]]
        interpretation['avg_category_activation'] = round(np.mean([n.activation for n in valid_category_nodes]), 4)
        if len(valid_category_nodes) > 1: interpretation['std_category_activation'] = round(np.std([n.activation for n in valid_category_nodes]), 4)
    else:
        interpretation['dominant_category'] = 'N/A'; interpretation['dominant_activation'] = 0.0; interpretation['category_ranking'] = []

    module_acts = {getattr(m, 'label', f'Mod_{i}'): round(float(getattr(m, 'activation', 0.0)), 4) for i, m in enumerate(module_nodes)}
    interpretation['module_activations'] = module_acts
    interpretation['value_node_activations'] = {v.label: round(float(v.activation), 4) for v in value_nodes if hasattr(v, 'label')}
    interpretation['emotion_state'] = CURRENT_EMOTION_STATE.copy()
    meta_cog = next((m for m in module_nodes if isinstance(m, MetaCognitio)), None);
    interpretation['last_reflection'] = meta_cog.reflection_log[-1] if meta_cog and meta_cog.reflection_log else None
    return interpretation

def generate_final_report(
    category_nodes: list, module_nodes: list, value_nodes: list,
    original_data: pd.DataFrame, interpretation_log: list
) -> tuple[str, dict]:
    """Generiert erweiterten Analysebericht."""
    print("\n--- Generiere finalen Bericht ---")
    report_lines = ["**NeuroPersona Erweiterter Analysebericht**\n"]
    structured_results = {
        "dominant_category": "N/A", "dominant_activation": 0.0, "category_ranking": [],
        "module_activations": {}, "value_node_activations": {}, "emotion_state": {},
        "final_assessment": "Keine klare Tendenz.", "frequent_dominant_category": None,
        "reflection_summary": [], "stability_assessment": "Unbekannt"
    }
    threshold_high = 0.65; threshold_low = 0.35

    # Kategorienanalyse (Final)
    report_lines.append("**Finale Netzwerk-Tendenzen (Kategorien):**")
    valid_category_nodes = [n for n in category_nodes if hasattr(n, 'activation') and isinstance(n.activation, (float, np.floating)) and not np.isnan(n.activation)]
    if not valid_category_nodes: report_lines.append("- Keine aktiven Kategorieknoten im finalen Zustand.")
    else:
        sorted_categories = sorted(valid_category_nodes, key=lambda n: n.activation, reverse=True)
        report_lines.append("  Aktivste Kategorien (Top 5):")
        category_ranking_data = []
        for i, node in enumerate(sorted_categories[:5]):
            label, act = getattr(node, 'label', 'Unb.'), round(node.activation, 3)
            report_lines.append(f"  {i+1}. {label}: {act}")
            category_ranking_data.append((label, act))
        structured_results["category_ranking"] = category_ranking_data
        structured_results["dominant_category"] = sorted_categories[0].label
        structured_results["dominant_activation"] = round(sorted_categories[0].activation, 4)

    # Kategorienanalyse (Verlauf)
    if interpretation_log:
        dom_cats_time = [e.get('dominant_category') for e in interpretation_log if e.get('dominant_category') != 'N/A']
        if dom_cats_time:
            try:
                most_freq, freq_count = Counter(dom_cats_time).most_common(1)[0]
                report_lines.append(f"- Verlauf: '{most_freq}' war am häufigsten dominant ({freq_count}/{len(interpretation_log)} Ep.).")
                structured_results["frequent_dominant_category"] = most_freq
                last_n = min(len(dom_cats_time), max(5, len(interpretation_log) // 4))
                recent_doms = dom_cats_time[-last_n:]
                unique_recent = len(set(recent_doms))
                if unique_recent == 1: stability = f"Stabil ('{recent_doms[0]}', letzte {last_n} Ep.)"
                elif unique_recent <= 2: stability = f"Wechselnd (zwischen {unique_recent} Kats., letzte {last_n} Ep.)"
                else: stability = f"Instabil ({unique_recent} Kats., letzte {last_n} Ep.)"
                structured_results["stability_assessment"] = stability
                report_lines.append(f"- Stabilität: {stability}")
            except IndexError: report_lines.append("- Verlauf: Keine dominante Kategorie gefunden.")
        else: report_lines.append("- Verlauf: Keine dominante Kategorie gefunden.")

    # Module (Final)
    report_lines.append("\n**Finaler Zustand der kognitiven Module:**")
    module_activation_data = {getattr(m, 'label', f'Mod_{i}'): round(getattr(m, 'activation', 0.0), 3) for i, m in enumerate(module_nodes)}
    sorted_modules = sorted(module_activation_data.items(), key=lambda item: item[1], reverse=True)
    for label, activation in sorted_modules: report_lines.append(f"- {label}: {activation}")
    structured_results["module_activations"] = module_activation_data

    # Werte (Final)
    report_lines.append("\n**Aktive Wertvorstellungen:**")
    value_activation_data = {v.label: round(v.activation, 3) for v in value_nodes if hasattr(v, 'label')}
    sorted_values = sorted(value_activation_data.items(), key=lambda item: item[1], reverse=True)
    for label, activation in sorted_values: report_lines.append(f"- {label}: {activation}")
    structured_results["value_node_activations"] = value_activation_data

    # Emotion (Final)
    report_lines.append("\n**Finale Emotionale Grundstimmung (PAD):**")
    limbus = next((m for m in module_nodes if isinstance(m, LimbusAffektus)), None)
    final_emotion_state = limbus.emotion_state if limbus else CURRENT_EMOTION_STATE
    for dim, value in final_emotion_state.items(): report_lines.append(f"- {dim.capitalize()}: {value:.3f}")
    structured_results["emotion_state"] = final_emotion_state

    # Meta-Kognition (Final)
    report_lines.append("\n**Meta-Kognitive Reflexion (Letzte Einträge):**")
    meta_cog = next((m for m in module_nodes if isinstance(m, MetaCognitio)), None)
    reflection_summary = []
    if meta_cog and meta_cog.reflection_log:
        logged_reflections = list(meta_cog.reflection_log)
        for i, entry in enumerate(reversed(logged_reflections)):
            if i >= 5: break
            msg, epoch_num = entry.get('message', ''), entry.get('epoch', '?')
            report_lines.append(f"- E{epoch_num}: {msg}")
            reflection_summary.append(entry)
    if not reflection_summary: report_lines.append("- Keine besonderen Vorkommnisse im Meta-Log.")
    structured_results["reflection_summary"] = reflection_summary # Immer speichern, auch wenn leer

    # Gesamteinschätzung
    final_assessment_text = structured_results.get("final_assessment", "Keine klare Tendenz.") # Holen aus structured_results (wird in run_neuropersona gesetzt)
    report_lines.append(f"\n**Gesamteinschätzung:** {final_assessment_text}") # Zeige die bereits ermittelte Einschätzung an

    final_report_text = "\n".join(report_lines)
    print(final_report_text)
    return final_report_text, structured_results


# --- Modell Speichern  ---
def save_final_network_state(nodes_list, emotion_state, value_nodes, meta_cog_module, filename=MODEL_FILENAME):
    """Speichert den finalen Zustand des Netzwerks in einer JSON-Datei."""
    model_data = {"nodes": [], "connections": [], "emotion_state": {}, "value_node_states": {}, "reflection_log": [], "version": "v_dare_complete_save_2"}
    valid_nodes = [node for node in nodes_list if hasattr(node, 'label') and isinstance(getattr(node,'label', None), str)]
    node_labels_set = {node.label for node in valid_nodes}
    print(f"Speichere Zustand von {len(valid_nodes)} Knoten in {filename}...")

    for node in valid_nodes:
        activation_raw = getattr(node, 'activation', 0.0)
        activation_save = float(activation_raw) if isinstance(activation_raw, (float, np.floating)) and not np.isnan(activation_raw) else 0.0
        node_info = {"label": node.label, "activation": round(activation_save, 5), "neuron_type": getattr(node, 'neuron_type', "excitatory"), "class": type(node).__name__}
        if isinstance(node, MemoryNode): node_info.update({"memory_type": getattr(node, 'memory_type', 'short_term'), "time_in_memory": getattr(node, 'time_in_memory', 0)})
        model_data["nodes"].append(node_info)
        if hasattr(node, 'connections'):
            for conn in node.connections:
                target_label = getattr(conn.target_node, 'label', None)
                if target_label and target_label in node_labels_set:
                    weight_raw = getattr(conn, 'weight', 0.0)
                    weight_save = float(weight_raw) if isinstance(weight_raw, (float, np.floating)) and not np.isnan(weight_raw) else 0.0
                    if weight_save > PRUNING_THRESHOLD * 0.5: # Nur signifikante Gewichte speichern?
                        model_data["connections"].append({"source": node.label, "target": target_label, "weight": round(weight_save, 5)})

    model_data["emotion_state"] = emotion_state
    model_data["value_node_states"] = {v.label: round(float(v.activation) if not np.isnan(v.activation) else 0.0, 5) for v in value_nodes if hasattr(v, 'label')}
    if meta_cog_module and hasattr(meta_cog_module, 'reflection_log'): model_data["reflection_log"] = list(meta_cog_module.reflection_log)

    try:
        with open(filename, "w", encoding='utf-8') as file: json.dump(model_data, file, indent=2, ensure_ascii=False)
        print(f"Netzwerkzustand gespeichert: {filename} ({len(model_data['connections'])} Verbindungen)")
    except (IOError, TypeError) as e: print(f"FEHLER beim Speichern des Zustands in '{filename}': {e}")


# --- Plotting ---
# (plot_activation_and_weights, plot_dynamics, plot_module_activation_comparison, plot_network_structure, plot_network_graph, plot_emotion_value_trends bleiben wie zuvor definiert)
# --- Plotting (leicht angepasste Dateinamen) ---
def filter_module_history(activation_history, module_labels):
    return {label: history for label, history in activation_history.items() if label in module_labels and history}

def plot_activation_and_weights(activation_history: dict, weights_history: dict, filename: str = "plot_act_weights.png") -> plt.Figure | None:
    print("Erstelle Plot: Aktivierungs- & Gewichtsentwicklung...")
    if not activation_history and not weights_history: return None
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8)); max_lines = 25
    valid_act_hist = {k: list(v) for k, v in activation_history.items() if isinstance(v, (list, deque)) and v}
    valid_weight_hist = {k: list(v) for k, v in weights_history.items() if isinstance(v, (list, deque)) and v}
    plot_count_act = 0
    if valid_act_hist:
        sorted_act_keys = sorted(valid_act_hist.keys(), key=lambda k: np.std(valid_act_hist[k]) if len(valid_act_hist[k]) > 1 else 0, reverse=True)
        for label in sorted_act_keys:
            if plot_count_act >= max_lines: break
            activations = valid_act_hist[label]
            if len(activations) > 1: ax1.plot(range(1, len(activations) + 1), activations, label=label, alpha=0.7, linewidth=1.5); plot_count_act += 1
    ax1.set_title(f"Aktivierungsentwicklung (Top {plot_count_act} dyn.)"); ax1.set_xlabel("Epoche"); ax1.set_ylabel("Aktivierung"); ax1.set_ylim(0, 1.05); ax1.grid(True, alpha=0.5)
    if plot_count_act > 0: ax1.legend(fontsize='small', loc='center left', bbox_to_anchor=(1.05, 0.5))
    plot_count_weights = 0
    if valid_weight_hist:
        sorted_weights_keys = sorted(valid_weight_hist.keys(), key=lambda k: np.std(valid_weight_hist[k]) if len(valid_weight_hist[k]) > 1 else 0, reverse=True)
        for label in sorted_weights_keys:
            if plot_count_weights >= max_lines: break
            weights = valid_weight_hist[label]
            if len(weights) > 1: ax2.plot(range(1, len(weights) + 1), weights, label=label, alpha=0.6, linewidth=1.0); plot_count_weights += 1
    ax2.set_title(f"Gewichtsentwicklung (Top {plot_count_weights} dyn.)"); ax2.set_xlabel("Epoche"); ax2.set_ylabel("Gewicht"); ax2.set_ylim(0, 1.05); ax2.grid(True, alpha=0.5)
    if plot_count_weights > 0: ax2.legend(fontsize='small', loc='center left', bbox_to_anchor=(1.05, 0.5))
    plt.tight_layout(rect=[0, 0, 0.88, 1])
    try:
        os.makedirs(PLOTS_FOLDER, exist_ok=True); filepath = os.path.join(PLOTS_FOLDER, filename)
        fig.savefig(filepath, bbox_inches='tight', dpi=100); print(f"Plot gespeichert: {filepath}"); return fig
    except Exception as e: print(f"FEHLER Plot '{filepath}': {e}"); return None
    finally: plt.close(fig)

def plot_dynamics(activation_history: dict, weights_history: dict, filename: str = "plot_dynamics.png") -> plt.Figure | None:
    print("Erstelle Plot: Netzwerk-Dynamiken...")
    if not activation_history and not weights_history: return None
    fig, axs = plt.subplots(2, 2, figsize=(16, 12), sharex=True); axs = axs.flatten()
    valid_act_hist = {k: list(v) for k, v in activation_history.items() if isinstance(v, (list, deque)) and v}
    valid_weight_hist = {k: list(v) for k, v in weights_history.items() if isinstance(v, (list, deque)) and v}
    num_epochs = max(max((len(h) for h in valid_act_hist.values()), default=0), max((len(h) for h in valid_weight_hist.values()), default=0))
    if num_epochs == 0: plt.close(fig); return None; epochs_range = np.arange(1, num_epochs + 1)
    def get_stats(hist_dict, ep_count):
        means, stds = [], [];
        for i in range(ep_count):
            vals = [h[i] for h in hist_dict.values() if len(h) > i and not np.isnan(h[i])];
            means.append(np.mean(vals) if vals else np.nan); stds.append(np.std(vals) if len(vals) > 1 else np.nan)
        return np.array(means), np.array(stds)
    avg_act, std_act = get_stats(valid_act_hist, num_epochs); valid_idx_act = ~np.isnan(avg_act)
    if np.any(valid_idx_act):
        axs[0].plot(epochs_range[valid_idx_act], avg_act[valid_idx_act], label="Avg. Akt.", color='blue'); axs[0].fill_between(epochs_range[valid_idx_act], np.maximum(0, avg_act[valid_idx_act]-std_act[valid_idx_act]), np.minimum(1, avg_act[valid_idx_act]+std_act[valid_idx_act]), alpha=0.2, color='blue', label="StdAbw"); axs[0].legend()
    axs[0].set_title("Netzwerkaktivierung"); axs[0].set_ylabel("Aktivierung"); axs[0].set_ylim(0, 1); axs[0].grid(True, alpha=0.5)
    avg_w, std_w = get_stats(valid_weight_hist, num_epochs); valid_idx_w = ~np.isnan(avg_w)
    if np.any(valid_idx_w):
        axs[1].plot(epochs_range[valid_idx_w], avg_w[valid_idx_w], label="Avg. Gewicht", color='green'); axs[1].fill_between(epochs_range[valid_idx_w], np.maximum(0, avg_w[valid_idx_w]-std_w[valid_idx_w]), np.minimum(1, avg_w[valid_idx_w]+std_w[valid_idx_w]), alpha=0.2, color='green', label="StdAbw"); axs[1].legend()
    axs[1].set_title("Gewichtsentwicklung"); axs[1].set_ylabel("Gewicht"); axs[1].set_ylim(0, max(1, np.nanmax(avg_w[valid_idx_w]+std_w[valid_idx_w]) if np.any(valid_idx_w) else 1)); axs[1].grid(True, alpha=0.5)
    active_nodes_count = []; total_nodes = len(valid_act_hist)
    if total_nodes > 0:
        for i in range(num_epochs): active_nodes_count.append(sum(1 for h in valid_act_hist.values() if len(h) > i and not np.isnan(h[i]) and h[i] > 0.5))
        if active_nodes_count: axs[2].plot(epochs_range, active_nodes_count, label="Aktive (>0.5)", color='red'); axs[2].set_ylim(0, total_nodes + 1); axs[2].legend()
    axs[2].set_title("Netzwerk-Aktivität"); axs[2].set_ylabel("Anzahl Knoten"); axs[2].grid(True, alpha=0.5)
    dom_act = []; module_labels = {m().label for m in (CortexCreativus, SimulatrixNeuralis, CortexCriticus, LimbusAffektus, MetaCognitio, CortexSocialis)}
    cat_hist = {k: h for k, h in valid_act_hist.items() if not k.startswith("Q_") and k not in module_labels}
    if cat_hist:
        for i in range(num_epochs): acts = [h[i] for h in cat_hist.values() if len(h) > i and not np.isnan(h[i])]; dom_act.append(max(acts) if acts else np.nan)
        dom_act_np = np.array(dom_act); valid_idx_dom = ~np.isnan(dom_act_np)
        if np.any(valid_idx_dom): axs[3].plot(epochs_range[valid_idx_dom], dom_act_np[valid_idx_dom], label="Max. Kat.-Akt.", color='purple'); axs[3].legend()
    axs[3].set_title("Dominanz Stärkste Kategorie"); axs[3].set_ylabel("Max. Aktivierung"); axs[3].set_ylim(0, 1); axs[3].grid(True, alpha=0.5)
    axs[3].set_xlabel("Epoche"); axs[2].set_xlabel("Epoche")
    plt.tight_layout()
    try:
        os.makedirs(PLOTS_FOLDER, exist_ok=True); filepath = os.path.join(PLOTS_FOLDER, filename)
        fig.savefig(filepath, bbox_inches='tight', dpi=100); print(f"Plot gespeichert: {filepath}"); return fig
    except Exception as e: print(f"FEHLER Plot '{filepath}': {e}"); return None
    finally: plt.close(fig)

def plot_module_activation_comparison(module_activation_history: dict, filename: str = "plot_modules.png") -> plt.Figure | None:
    print("Erstelle Plot: Modul-Aktivierungsvergleich...")
    if not module_activation_history: return None
    fig, ax = plt.subplots(figsize=(12, 7)); plotted = False
    for label, activations_deque in module_activation_history.items():
        activations = list(activations_deque)
        if len(activations) > 1: ax.plot(range(1, len(activations) + 1), activations, label=label, linewidth=2, alpha=0.8); plotted = True
    if not plotted: plt.close(fig); return None
    ax.set_title("Vergleich Aktivierungen Kognitiver Module"); ax.set_xlabel("Epoche"); ax.set_ylabel("Aktivierung"); ax.set_ylim(0, 1.05); ax.grid(True, alpha=0.5); ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5)); plt.tight_layout(rect=[0, 0, 0.85, 1])
    try:
        os.makedirs(PLOTS_FOLDER, exist_ok=True); filepath = os.path.join(PLOTS_FOLDER, filename)
        fig.savefig(filepath, bbox_inches='tight', dpi=100); print(f"Plot gespeichert: {filepath}"); return fig
    except Exception as e: print(f"FEHLER Plot '{filepath}': {e}"); return None
    finally: plt.close(fig)

def plot_network_structure(nodes_list: list, filename: str = "plot_structure_stats.png") -> plt.Figure | None:
    print("Erstelle Plot: Netzwerk-Struktur (Statistiken)...")
    if not nodes_list: return None
    node_counts_by_type = Counter(); neuron_type_counts = Counter(); connection_weights = []; total_connections = 0; module_classes = (CortexCreativus, SimulatrixNeuralis, CortexCriticus, LimbusAffektus, MetaCognitio, CortexSocialis)
    for node in nodes_list:
        node_type = "Andere"; label = getattr(node, 'label', '')
        if isinstance(node, module_classes): node_type = f"Modul: {label}"
        elif isinstance(node, ValueNode): node_type = 'Wert'
        elif isinstance(node, MemoryNode): node_type = 'Kategorie (Memory)'
        elif label.startswith("Q_"): node_type = 'Frage (Input)'
        elif isinstance(node, Node): node_type = 'Basis-Knoten'
        node_counts_by_type[node_type] += 1
        neuron_type_counts[getattr(node, 'neuron_type', 'unbekannt')] += 1
        if hasattr(node, 'connections'):
            valid_conns = [conn for conn in node.connections if hasattr(conn, 'weight') and isinstance(conn.weight, (float, np.floating)) and not np.isnan(conn.weight)]
            total_connections += len(valid_conns); connection_weights.extend([float(conn.weight) for conn in valid_conns])
    num_bars = len(node_counts_by_type); fig_width = max(15, 7 + num_bars * 0.8)
    fig, axs = plt.subplots(1, 3, figsize=(fig_width, 6), gridspec_kw={'width_ratios': [max(3, num_bars*0.4), 1, 1.5]})
    if node_counts_by_type:
        types = sorted(node_counts_by_type.keys()); counts = [node_counts_by_type[t] for t in types]; axs[0].bar(types, counts, color='skyblue'); axs[0].set_title(f'Knotentypen (Gesamt: {len(nodes_list)})'); axs[0].set_ylabel('Anzahl'); axs[0].tick_params(axis='x', rotation=45, ha='right', labelsize='small'); axs[0].grid(True, axis='y', alpha=0.6)
    if neuron_type_counts:
        n_types = sorted(neuron_type_counts.keys()); n_counts = [neuron_type_counts[nt] for nt in n_types]; axs[1].bar(n_types, n_counts, color='lightcoral'); axs[1].set_title('Neuronentypen'); axs[1].set_ylabel('Anzahl'); axs[1].grid(True, axis='y', alpha=0.6)
    if connection_weights:
        axs[2].hist(connection_weights, bins=25, color='lightgreen', edgecolor='black', alpha=0.7); avg_w = np.mean(connection_weights); std_w = np.std(connection_weights); axs[2].axvline(avg_w, color='red', ls='--', lw=1.5, label=f'Avg: {avg_w:.3f}'); axs[2].set_title(f'Gewichtsverteilung (N={total_connections})'); axs[2].set_xlabel('Gewicht'); axs[2].set_ylabel('Häufigkeit'); axs[2].legend(fontsize='small'); axs[2].grid(True, axis='y', alpha=0.6); stats_text = f'StdAbw: {std_w:.3f}\nMin: {min(connection_weights):.3f}\nMax: {max(connection_weights):.3f}'; axs[2].text(0.95, 0.95, stats_text, transform=axs[2].transAxes, fontsize='small', va='top', ha='right', bbox=dict(boxstyle='round', fc='wheat', alpha=0.5))
    else: axs[2].text(0.5, 0.5, 'Keine Verbindungen', ha='center'); axs[2].set_title('Gewichtsverteilung (N=0)')
    plt.tight_layout(pad=2.0)
    try:
        os.makedirs(PLOTS_FOLDER, exist_ok=True); filepath = os.path.join(PLOTS_FOLDER, filename)
        fig.savefig(filepath, bbox_inches='tight', dpi=100); print(f"Plot gespeichert: {filepath}"); return fig
    except Exception as e: print(f"FEHLER Plot '{filepath}': {e}"); return None
    finally: plt.close(fig)

def plot_network_graph(nodes_list: list, filename: str = "plot_network_graph.png") -> plt.Figure | None:
    print("Erstelle Plot: Netzwerk-Graph...")
    if not NETWORKX_AVAILABLE: print("Übersprungen: networkx fehlt."); return None
    if not nodes_list: print("Keine Knotendaten für Graph."); return None
    G = nx.DiGraph(); node_labels = {}; node_info = {}; edge_weights = {}; module_classes = (CortexCreativus, SimulatrixNeuralis, CortexCriticus, LimbusAffektus, MetaCognitio, CortexSocialis); valid_node_labels = set()
    for node in nodes_list:
        if hasattr(node, 'label') and isinstance(node.label, str):
            label = node.label
            if label not in G: G.add_node(label); node_labels[label] = label; valid_node_labels.add(label); act = float(getattr(node, 'activation', 0.0)); node_type = "base";
            if isinstance(node, module_classes): node_type = "module"
            elif isinstance(node, ValueNode): node_type = "value"
            elif isinstance(node, MemoryNode): node_type = "memory"
            elif label.startswith("Q_"): node_type = "question"
            node_info[label] = {"type": node_type, "activation": act}
            if hasattr(node, 'connections'):
                for conn in node.connections:
                    target_node = conn.target_node
                    if hasattr(target_node, 'label') and isinstance(target_node.label, str):
                        target_label = target_node.label
                        if target_label in valid_node_labels: # Ziel muss auch gültig sein
                            weight = float(getattr(conn, 'weight', 0.0))
                            if weight > 0.05: G.add_edge(label, target_label, weight=weight); edge_weights[(label, target_label)] = f"{weight:.2f}"
    if not G.nodes: print("Keine Knoten im Graphen nach Filterung."); return None
    node_colors, node_sizes = [], []; nodes_to_draw = list(G.nodes())
    for node_label in nodes_to_draw:
        info = node_info.get(node_label, {"type": "unknown", "activation": 0.0}); node_type, act = info["type"], info["activation"]; color = 'grey'; size = 300 + act * 1500
        if node_type == "module": color = 'red'; size *= 1.5
        elif node_type == "value": color = 'gold'; size *= 1.2
        elif node_type == "memory": color = 'skyblue'
        elif node_type == "question": color = 'lightgreen'; size *= 0.8
        node_colors.append(color); node_sizes.append(max(100, size))
    fig, ax = plt.subplots(figsize=(22, 18))
    try:
        pos = nx.kamada_kawai_layout(G) if len(G.nodes) < 150 else nx.spring_layout(G, k=0.5/np.sqrt(len(G.nodes)), iterations=50, seed=42)
        nx.draw_networkx_nodes(G, pos, nodelist=nodes_to_draw, node_size=node_sizes, node_color=node_colors, alpha=0.85, ax=ax)
        nx.draw_networkx_edges(G, pos, width=0.6, alpha=0.3, edge_color='gray', style='solid', arrows=True, arrowstyle='-|>', arrowsize=10, node_size=node_sizes, ax=ax)
        labels_to_draw = {n: n for n in nodes_to_draw if node_info[n]['type'] in ['module', 'value'] or node_info[n]['activation'] > 0.5}
        nx.draw_networkx_labels(G, pos, labels=labels_to_draw, font_size=7, ax=ax)
        # nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_weights, font_size=5, font_color='darkred', ax=ax) # Zu unübersichtlich
        ax.set_title(f"Netzwerk Graph (Nodes: {len(G.nodes)}, Edges: {len(G.edges)})", fontsize=16); plt.axis('off'); plt.tight_layout()
        os.makedirs(PLOTS_FOLDER, exist_ok=True); filepath = os.path.join(PLOTS_FOLDER, filename); fig.savefig(filepath, bbox_inches='tight', dpi=120); print(f"Plot gespeichert: {filepath}"); return fig
    except Exception as e: print(f"FEHLER Graph Plot '{filename}': {e}"); import traceback; traceback.print_exc(); return None
    finally: plt.close(fig)

def plot_emotion_value_trends(interpretation_log: list, value_history: dict, filename: str = "plot_emo_values.png") -> plt.Figure | None:
    print("Erstelle Plot: Emotions- & Werte-Trends...")
    if not interpretation_log: return None
    epochs = [log.get('epoch', i + 1) for i, log in enumerate(interpretation_log)]
    if not epochs: return None
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), sharex=True)
    pleasure = [log.get('emotion_state', {}).get('pleasure', np.nan) for log in interpretation_log]
    arousal = [log.get('emotion_state', {}).get('arousal', np.nan) for log in interpretation_log]
    dominance = [log.get('emotion_state', {}).get('dominance', np.nan) for log in interpretation_log]
    ax1.plot(epochs, pleasure, label='Pleasure', color='green', alpha=0.8, marker='.', markersize=3, linestyle='-'); ax1.plot(epochs, arousal, label='Arousal', color='red', alpha=0.8, marker='.', markersize=3, linestyle='-'); ax1.plot(epochs, dominance, label='Dominance', color='blue', alpha=0.8, marker='.', markersize=3, linestyle='-'); ax1.set_title('Emotionsverlauf (PAD)'); ax1.set_ylabel('Level [-1, 1]'); ax1.set_ylim(-1.1, 1.1); ax1.legend(); ax1.grid(True, alpha=0.5); ax1.axhline(0, color='grey', ls='--', lw=0.7)
    plotted_values = False
    if value_history:
        min_len = min((len(h) for h in value_history.values() if h), default=0); effective_epochs = epochs[:min_len] if min_len > 0 else []
        if effective_epochs:
            for v_label, history in value_history.items():
                 if history: ax2.plot(effective_epochs, list(history)[:min_len], label=v_label, alpha=0.7, marker='.', markersize=3, linestyle='-'); plotted_values = True
    if plotted_values: ax2.set_title('Werteverlauf'); ax2.set_ylabel('Aktivierung [0, 1]'); ax2.set_ylim(-0.05, 1.05); ax2.legend(fontsize='small', loc='center left', bbox_to_anchor=(1.02, 0.5)); ax2.grid(True, alpha=0.5)
    else: ax2.set_title('Werteverlauf (Keine Daten)')
    ax2.set_xlabel('Epoche'); plt.tight_layout(rect=[0, 0, 0.9, 1])
    try:
        os.makedirs(PLOTS_FOLDER, exist_ok=True); filepath = os.path.join(PLOTS_FOLDER, filename)
        fig.savefig(filepath, bbox_inches='tight', dpi=100); print(f"Plot gespeichert: {filepath}"); return fig
    except Exception as e: print(f"FEHLER Plot '{filepath}': {e}"); return None
    finally: plt.close(fig)

# --- Wichtige Kategorien ---
def get_important_categories(category_nodes: list, top_n: int = 5) -> List[Tuple[str, str]]:
    """Ermittelt die wichtigsten Kategorien basierend auf finaler Aktivierung."""
    valid_nodes = [n for n in category_nodes if hasattr(n, 'label') and hasattr(n, 'activation') and isinstance(getattr(n, 'activation', None), (float, np.floating)) and not np.isnan(n.activation)]
    valid_nodes.sort(key=lambda n: n.activation, reverse=True)
    important_categories = []
    for node in valid_nodes[:top_n]:
        act = node.activation; importance = ("sehr hoch" if act >= 0.8 else "hoch" if act >= 0.65 else "mittel" if act >= 0.4 else "gering" if act >= 0.2 else "sehr gering")
        important_categories.append((node.label, importance))
    return important_categories


# --- Hauptfunktion ---
def run_neuropersona_simulation(
    input_df: pd.DataFrame,
    epochs: int = DEFAULT_EPOCHS,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    decay_rate: float = DEFAULT_DECAY_RATE,
    reward_interval: int = DEFAULT_REWARD_INTERVAL,
    generate_plots: bool = True,
    save_state: bool = False,
    load_state: bool = False,
    status_callback: Callable[[str], None] = _default_status_callback
) -> tuple[str | None, dict | None]:
    """Führt die NeuroPersona-Simulation durch."""
    start_time = time.time()
    status_callback("\n--- Starte Erweiterte NeuroPersona Simulation ---")
    status_callback(f"Params: E={epochs}, LR={learning_rate:.4f}, DR={decay_rate:.4f}, RI={reward_interval}, Load={load_state}, Save={save_state}, Plots={generate_plots}")

    # 1. Validierung & Vorverarbeitung
    if not isinstance(input_df, pd.DataFrame) or input_df.empty: return None, {"error": "Leeres Input-DataFrame."}
    required_cols = ['Frage', 'Antwort', 'Kategorie']
    if not all(col in input_df.columns for col in required_cols): return None, {"error": f"Fehlende Spalten: {[c for c in required_cols if c not in input_df.columns]}"}
    processed_data = preprocess_data(input_df)
    if processed_data.empty or 'Kategorie' not in processed_data.columns or processed_data['Kategorie'].nunique() == 0: return None, {"error": "Datenvorverarbeitung fehlgeschlagen."}

    # 2. Persistenz-Setup (verwende globale Instanz)
    global persistent_memory_manager
    if persistent_memory_manager is None:
        try: persistent_memory_manager = PersistentMemoryManager(); status_callback("Persistent Memory Manager initialisiert.")
        except Exception as e: status_callback(f"WARNUNG: Init Persistent Memory fehlgeschlagen: {e}"); persistent_memory_manager = None

    # 3. Netzwerkaufbau
    categories = processed_data['Kategorie'].unique()
    category_nodes, module_nodes, value_nodes = initialize_network_nodes(categories)
    if not category_nodes or not module_nodes or not value_nodes: return None, {"error": "Netzwerkinitialisierung fehlgeschlagen."}

    # 4. Simulation
    activation_history_final, weights_history_final, interpretation_log = {}, {}, []
    final_category_nodes, final_module_nodes, final_value_nodes = [], [], []
    all_final_nodes, final_value_history = [], {}
    load_filename = MODEL_FILENAME if load_state else None
    simulation_successful = False
    try:
        sim_results = simulate_learning_cycle(
            processed_data, category_nodes, module_nodes, value_nodes, epochs=epochs,
            learning_rate=learning_rate, reward_interval=reward_interval, decay_rate=decay_rate,
            initial_emotion_state=INITIAL_EMOTION_STATE.copy(), persistent_memory=persistent_memory_manager,
            load_state_from=load_filename, status_callback=status_callback
        )
        activation_history_final, weights_history_final, interpretation_log, \
        final_category_nodes, final_module_nodes, final_value_nodes, \
        all_final_nodes, final_value_history = sim_results
        simulation_successful = True
    except Exception as e:
        status_callback(f"FATALER FEHLER im Simulationszyklus: {e}")
        print(f"FATALER FEHLER: {e}"); import traceback; traceback.print_exc()
        return None, {"error": f"Simulationsfehler: {e}", "partial_log": interpretation_log[-5:]}

    if not simulation_successful or (not final_category_nodes and not final_module_nodes):
        status_callback("WARNUNG: Simulation unvollständig oder keine finalen Knoten.")
        if interpretation_log:
             final_report_text, structured_results = generate_final_report([], [], [], processed_data, interpretation_log)
             structured_results["warning"] = "Simulation unvollständig, keine finalen Knoten."
             return final_report_text, structured_results
        else: return "Simulation fehlgeschlagen.", {"error": "Keine finalen Knoten und Logs."}

    # 5. Abschlussbericht
    status_callback("Generiere finalen Bericht...")
    final_report_text, structured_results = generate_final_report(
        final_category_nodes, final_module_nodes, final_value_nodes,
        processed_data, interpretation_log
    )

    # 6. Plots (optional)
    if generate_plots:
        status_callback("\n--- Generiere Plots ---")
        os.makedirs(PLOTS_FOLDER, exist_ok=True); plot_errors = []
        try:
            plot_activation_and_weights(activation_history_final, weights_history_final)
            plot_dynamics(activation_history_final, weights_history_final)
            module_labels = [m.label for m in final_module_nodes if hasattr(m, 'label')]
            module_hist = filter_module_history(activation_history_final, module_labels)
            plot_module_activation_comparison(module_hist)
            plot_network_structure(all_final_nodes)
            if NETWORKX_AVAILABLE: plot_network_graph(all_final_nodes)
            plot_emotion_value_trends(interpretation_log, final_value_history)
            plt.close('all'); status_callback("Plots generiert.")
        except Exception as plot_error:
            status_callback(f"FEHLER Plot-Generierung: {plot_error}"); print(f"FEHLER Plots: {plot_error}")
            plot_errors.append(str(plot_error)); plt.close('all')
        if plot_errors: structured_results["plot_errors"] = plot_errors

    # 7. Empfehlung
    dom_cat = structured_results.get("dominant_category", "N/A"); dom_act = structured_results.get("dominant_activation", 0.0)
    pleasure = structured_results.get("emotion_state", {}).get('pleasure', 0.0); stability = structured_results.get("stability_assessment", "Unbekannt")
    final_recommendation = "Abwarten"; pos_kws = ["chance", "wachstum", "positiv", "potential", "innovation", "lösung"]; neg_kws = ["risiko", "problem", "negativ", "bedrohung", "schwierigkeit"]
    if dom_cat != "N/A" and dom_act > 0.5:
        cat_low = dom_cat.lower(); is_pos = any(kw in cat_low for kw in pos_kws); is_neg = any(kw in cat_low for kw in neg_kws)
        if is_pos and not is_neg: final_recommendation = "Empfehlung (moderat)" if dom_act < 0.75 or pleasure < 0.1 or "Stabil" not in stability else "Empfehlung"
        elif is_neg and not is_pos: final_recommendation = "Abraten (moderat)" if dom_act < 0.7 or pleasure > -0.1 or "Stabil" not in stability else "Abraten"
        elif dom_act < 0.4 or "Instabil" in stability: final_recommendation = "Abwarten (Instabil/Schwach)"
    structured_results["final_recommendation"] = final_recommendation

    # 8. Wichtige Kategorien & HTML-Report
    important_categories = get_important_categories(final_category_nodes, top_n=5)
    html_filename = f"neuropersona_report_{time.strftime('%Y%m%d_%H%M%S')}.html"
    create_html_report(final_report_text, final_recommendation, interpretation_log, important_categories, structured_results, PLOTS_FOLDER, html_filename)

    # 9. Zustand speichern (optional)
    if save_state:
        if all_final_nodes:
            meta_cog = next((m for m in final_module_nodes if isinstance(m, MetaCognitio)), None)
            save_final_network_state(all_final_nodes, CURRENT_EMOTION_STATE, final_value_nodes, meta_cog, MODEL_FILENAME)
        else: status_callback("Info: Speichern übersprungen, keine finalen Knoten.")

    end_time = time.time()
    exec_time = round(end_time - start_time, 2)
    status_callback(f"--- NeuroPersona Simulation abgeschlossen ({exec_time:.2f}s) ---")
    structured_results["execution_time_seconds"] = exec_time

    # 10. DB Verbindung schließen (WICHTIG!)
    if persistent_memory_manager:
        persistent_memory_manager.close()

    return final_report_text, structured_results


# --- GUI Code ---
def start_gui():
    """Startet die GUI für den NeuroPersona Workflow."""
    root = tk.Tk()
    root.title("NeuroPersona Workflow Starter (v_dare_complete)")
    root.geometry("520x450")

    style = ttk.Style()
    style.theme_use('clam')

    main_frame = ttk.Frame(root, padding="15")
    main_frame.pack(fill=tk.BOTH, expand=True)

    param_container = ttk.LabelFrame(main_frame, text="Simulationsparameter", padding="10")
    param_container.pack(fill=tk.X, padx=5, pady=(5, 10))
    param_container.columnconfigure(1, weight=1)

    entry_widgets = {}
    params = [("Lernrate:", 'learning_rate', DEFAULT_LEARNING_RATE), ("Decay Rate:", 'decay_rate', DEFAULT_DECAY_RATE), ("Reward Interval:", 'reward_interval', DEFAULT_REWARD_INTERVAL), ("Epochen:", 'epochs', DEFAULT_EPOCHS)]
    for row_idx, (label_text, key, default_value) in enumerate(params):
        ttk.Label(param_container, text=label_text).grid(row=row_idx, column=0, sticky=tk.W, pady=4, padx=5)
        entry = ttk.Entry(param_container, width=12)
        entry.insert(0, str(default_value))
        entry.grid(row=row_idx, column=1, sticky=tk.EW, pady=4, padx=5)
        entry_widgets[key] = entry

    options_container = ttk.Frame(param_container)
    options_container.grid(row=len(params), column=0, columnspan=2, sticky=tk.W, pady=(10, 0))
    generate_plots_var = tk.BooleanVar(value=True); save_state_var = tk.BooleanVar(value=False); load_state_var = tk.BooleanVar(value=False)
    ttk.Checkbutton(options_container, text="Plots", variable=generate_plots_var).pack(side=tk.LEFT, padx=(0,10))
    ttk.Checkbutton(options_container, text="Speichern", variable=save_state_var).pack(side=tk.LEFT, padx=10)
    ttk.Checkbutton(options_container, text="Laden", variable=load_state_var).pack(side=tk.LEFT, padx=10)

    prompt_container = ttk.LabelFrame(main_frame, text="Analyse-Anfrage / Thema", padding="10")
    prompt_container.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
    user_prompt_text = scrolledtext.ScrolledText(prompt_container, height=6, width=50, wrap=tk.WORD, relief=tk.SOLID, borderwidth=1, font=("Segoe UI", 9))
    user_prompt_text.pack(fill=tk.BOTH, expand=True, pady=5)
    user_prompt_text.insert("1.0", "Geben Sie hier Ihre Frage oder das zu analysierende Thema ein...\nz.B. Analyse der Chancen und Risiken von generativer KI im Bildungssektor.")

    status_label = ttk.Label(main_frame, text="Status: Bereit", anchor=tk.W, relief=tk.GROOVE, padding=(5, 2))
    status_label.pack(fill=tk.X, padx=5, pady=5)

    button_frame = ttk.Frame(main_frame)
    button_frame.pack(fill=tk.X, padx=5, pady=(5, 10))
    button_frame.columnconfigure(0, weight=1)
    button_frame.columnconfigure(3, weight=1)

    def save_gui_settings():
        settings_data = {"basic_params": {name: widget.get() for name, widget in entry_widgets.items()}, "options": {"generate_plots": generate_plots_var.get(), "save_state": save_state_var.get(), "load_state": load_state_var.get()}}
        filepath = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("JSON files", "*.json")], initialfile=SETTINGS_FILENAME, title="Parameter speichern")
        if not filepath: status_label.config(text="Status: Speichern abgebrochen."); return
        try:
            with open(filepath, 'w', encoding='utf-8') as f: json.dump(settings_data, f, indent=2)
            status_label.config(text=f"Status: Parameter gespeichert: {os.path.basename(filepath)}")
        except Exception as e: messagebox.showerror("Fehler", f"Speichern fehlgeschlagen:\n{e}"); status_label.config(text="Status: Fehler beim Speichern.")

    def load_gui_settings():
        filepath = filedialog.askopenfilename(defaultextension=".json", filetypes=[("JSON files", "*.json")], initialfile=SETTINGS_FILENAME, title="Parameter laden")
        if not filepath or not os.path.exists(filepath): status_label.config(text="Status: Laden abgebrochen/Datei fehlt."); return
        try:
            with open(filepath, 'r', encoding='utf-8') as f: settings_data = json.load(f)
            for name, widget in entry_widgets.items():
                if name in settings_data.get("basic_params", {}): widget.delete(0, tk.END); widget.insert(0, str(settings_data["basic_params"][name]))
            options = settings_data.get("options", {}); generate_plots_var.set(options.get("generate_plots", True)); save_state_var.set(options.get("save_state", False)); load_state_var.set(options.get("load_state", False))
            status_label.config(text=f"Status: Parameter geladen: {os.path.basename(filepath)}")
        except Exception as e: messagebox.showerror("Fehler", f"Laden fehlgeschlagen:\n{e}"); status_label.config(text="Status: Fehler beim Laden.")

    def display_final_result(result_text: str, parent_root):
        """Zeigt das finale Ergebnis in einem neuen Fenster."""
        if not parent_root.winfo_exists(): return # Nichts anzeigen, wenn Hauptfenster geschlossen
        result_window = tk.Toplevel(parent_root)
        result_window.title("NeuroPersona Workflow Ergebnis")
        result_window.geometry("750x550")
        result_window.transient(parent_root) # Bleibt über dem Hauptfenster
        result_window.grab_set() # Modal machen

        st_widget = scrolledtext.ScrolledText(result_window, wrap=tk.WORD, padx=10, pady=10, relief=tk.FLAT, font=("Segoe UI", 9))
        st_widget.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        st_widget.insert(tk.END, result_text if result_text else "Kein Ergebnis zurückgegeben.")
        st_widget.configure(state='disabled') # Read-only

        button_bar = ttk.Frame(result_window)
        button_bar.pack(pady=(0, 10))
        ttk.Button(button_bar, text="Schließen", command=result_window.destroy).pack()

        result_window.wait_window() # Warte bis dieses Fenster geschlossen wird

    def run_workflow_in_thread(user_prompt, lr, dr, ri, ep, gen_plots, save_st, load_st, status_cb_ref, start_btn_ref, save_btn_ref, load_btn_ref, root_ref):
        """Führt den Workflow im Hintergrund aus und aktualisiert die GUI."""
        final_result_text = "Workflow gestartet..."
        try:
            # Dynamischer Import des Orchestrators
            try:
                 orchestrator_module = importlib.import_module("orchestrator")
                 execute_full_workflow = getattr(orchestrator_module, "execute_full_workflow")
                 status_cb_ref("Orchestrator geladen.")
            except (ImportError, AttributeError) as import_err:
                error_msg = f"Orchestrator nicht gefunden oder fehlerhaft:\n{import_err}\nStellen Sie sicher, dass 'orchestrator.py' im selben Verzeichnis liegt und die Funktion 'execute_full_workflow' enthält."
                status_cb_ref("Fehler: Orchestrator fehlt!")
                # Zeige Fehler im Hauptthread
                if root_ref.winfo_exists():
                    root_ref.after(0, lambda: messagebox.showerror("Modul Fehler", error_msg))
                return # Beende Thread

            # Führe den eigentlichen Workflow aus
            status_cb_ref("Führe Workflow aus...")
            final_result_text = execute_full_workflow(
                user_prompt,
                neuropersona_epochs=ep,
                neuropersona_lr=lr,
                neuropersona_dr=dr,
                neuropersona_ri=ri,
                neuropersona_gen_plots=gen_plots,
                neuropersona_save_state=save_st,
                neuropersona_load_state=load_st,
                status_callback=status_cb_ref # Übergib die GUI-Update-Funktion
            )

            # Status nach Workflow-Ende
            if final_result_text and isinstance(final_result_text, str) and "FEHLER" in final_result_text.upper():
                status_cb_ref("Workflow mit Fehlern beendet.")
            elif final_result_text:
                status_cb_ref("Workflow erfolgreich abgeschlossen.")
            else:
                status_cb_ref("Workflow beendet, kein Ergebnis zurückgegeben.")


            # Zeige Ergebnis im Hauptthread an (wenn vorhanden)
            if final_result_text and root_ref.winfo_exists():
                 root_ref.after(0, lambda: display_final_result(final_result_text, root_ref))

        except Exception as e:
            error_traceback = traceback.format_exc()
            print(f"FATALER FEHLER im Workflow-Thread: {e}\n{error_traceback}")
            status_cb_ref(f"Schwerwiegender Fehler: {e}")
            # Zeige Fehler im Hauptthread
            if root_ref.winfo_exists():
                root_ref.after(0, lambda: messagebox.showerror("Workflow Fehler", f"Unerwarteter Fehler im Workflow:\n{e}\n\nDetails siehe Konsole."))
        finally:
            # Aktiviere Buttons wieder im Hauptthread, egal was passiert ist
            if root_ref.winfo_exists():
                root_ref.after(0, lambda: start_btn_ref.config(state=tk.NORMAL))
                root_ref.after(0, lambda: save_btn_ref.config(state=tk.NORMAL))
                root_ref.after(0, lambda: load_btn_ref.config(state=tk.NORMAL))
            # Schließe die DB-Verbindung auch hier, falls sie noch offen ist
            if persistent_memory_manager:
                persistent_memory_manager.close()

    def start_full_workflow_action():
        user_prompt = user_prompt_text.get("1.0", tk.END).strip()
        if not user_prompt or user_prompt.startswith("Geben Sie hier"):
            messagebox.showwarning("Eingabe fehlt", "Bitte geben Sie eine Analyse-Anfrage oder ein Thema ein.")
            return
        try:
            lr = float(entry_widgets['learning_rate'].get().replace(',', '.'))
            dr = float(entry_widgets['decay_rate'].get().replace(',', '.'))
            ri = int(entry_widgets['reward_interval'].get())
            ep = int(entry_widgets['epochs'].get())
            if not (0 < lr <= 1.0 and 0 <= dr < 1.0 and ri >= 1 and ep >= 1): raise ValueError("Ungültige Parameterbereiche.")
        except ValueError as ve: messagebox.showerror("Eingabefehler", f"Ungültiger Parameterwert.\n({ve})"); return

        status_label.config(text="Status: Starte Workflow...")
        start_button.config(state=tk.DISABLED); save_button.config(state=tk.DISABLED); load_button.config(state=tk.DISABLED)
        def gui_status_update(message: str):
            if root.winfo_exists(): root.after(0, lambda: status_label.config(text=f"Status: {message[:100]}..."))
        threading.Thread(target=run_workflow_in_thread, args=(user_prompt, lr, dr, ri, ep, generate_plots_var.get(), save_state_var.get(), load_state_var.get(), gui_status_update, start_button, save_button, load_button, root), daemon=True).start()

    # Buttons erstellen und platzieren
    save_button = ttk.Button(button_frame, text="Params Speichern", command=save_gui_settings)
    save_button.grid(row=0, column=1, padx=5)
    load_button = ttk.Button(button_frame, text="Params Laden", command=load_gui_settings)
    load_button.grid(row=0, column=2, padx=5)
    start_button = ttk.Button(button_frame, text="Workflow starten", style="Accent.TButton", command=start_full_workflow_action)
    start_button.grid(row=0, column=3, padx=(15, 0))

    style.configure("Accent.TButton", font=("Segoe UI", 10, "bold"), foreground="white", background="#007bff")
    style.map("Accent.TButton", background=[('active', '#0056b3')]) # Hover-Effekt

    # Frühere Einstellungen laden (optional)
    if os.path.exists(SETTINGS_FILENAME):
        try:
             load_gui_settings()
             status_label.config(text=f"Status: Gespeicherte Parameter '{SETTINGS_FILENAME}' geladen.")
        except Exception as e:
             status_label.config(text=f"Status: Fehler beim Laden von '{SETTINGS_FILENAME}'.")


    # Stelle sicher, dass die DB-Verbindung beim Schließen der GUI geschlossen wird
    def on_closing():
        if messagebox.askokcancel("Beenden", "NeuroPersona Workflow Starter beenden?"):
            print("GUI wird geschlossen.")
            if persistent_memory_manager:
                persistent_memory_manager.close()
            root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()


if __name__ == "__main__":
    # Initialisiere den Persistent Memory Manager einmal global
    try:
        persistent_memory_manager = PersistentMemoryManager()
    except Exception as e:
        print(f"FEHLER bei globaler Initialisierung von PersistentMemoryManager: {e}")
        persistent_memory_manager = None # Stelle sicher, dass es None ist

    start_gui()

    # Optional: Stelle sicher, dass die DB-Verbindung auch nach GUI-Schließung geschlossen wird
    # (wird jetzt in on_closing und im finally-Block des Threads gemacht)
    # if persistent_memory_manager:
    #    persistent_memory_manager.close()
    print("NeuroPersona Core beendet.")
