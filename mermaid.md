## Test

```mermaid
graph TD
    %% =============================================
    %% == 1. GUI & Initialisierung               ==
    %% =============================================
    StartApp["User startet neuropersona_core.py<br>(if __name__ == '__main__')"] --> CallStartGUI[rufe start_gui()]
    CallStartGUI --> InitGUI[Initialisiere Tkinter GUI<br>(Fenster, Widgets, Parameter, Optionen)]
    InitGUI --> LoadSettingsCheck{"Existiert<br>settings.json?"}
    LoadSettingsCheck -- Ja --> CallLoadGUISettings[rufe load_gui_settings()] --> UpdateGUIWidgets[Aktualisiere GUI-Widgets<br>aus JSON] --> GUIReady
    LoadSettingsCheck -- Nein --> GUIReady[GUI bereit für Interaktion]

    GUIReady --> UserInteraction["User Interaktion:<br>- Ändert Parameter<br>- Gibt Prompt ein<br>- Klickt Buttons"]
    UserInteraction -- Klick 'Params speichern' --> CallSaveGUISettings[rufe save_gui_settings()] --> WriteJSON[Schreibe GUI-State<br>nach JSON] --> GUIReady
    UserInteraction -- Klick 'Params laden' --> CallLoadGUISettings
    UserInteraction -- Klick 'Workflow starten' --> CallStartWorkflowAction[rufe start_full_workflow_action()]

    CallStartWorkflowAction --> GetGUIInput["Hole User-Prompt<br>& Parameter aus GUI"]
    GetGUIInput --> ValidateInput{"Parameter &<br>Prompt gültig?"}
    ValidateInput -- Nein --> ShowErrorMsg[Zeige Fehlermeldung<br>in GUI-Statusleiste] --> GUIReady
    ValidateInput -- Ja --> DisableStartButton[Deaktiviere Start-Button]
    DisableStartButton --> CreateWorkflowThread["Erstelle & Starte neuen Thread<br>für run_workflow_in_thread<br>(args: prompt, params, flags, callback)"]
    CreateWorkflowThread --> UpdateGUIStatusInit[Update GUI Status:<br>'Starte Workflow...']
    CreateWorkflowThread --> WFThreadStart(Workflow Thread Start)


    %% =============================================
    %% == 2. Workflow Thread & Orchestrator       ==
    %% =============================================
    subgraph Workflow_Thread [Workflow-Thread: run_workflow_in_thread]
        direction TB
        WFThreadStart --> ImportOrchestrator{"Importiere<br>orchestrator Modul?"}
        ImportOrchestrator -- Nein --> WF_ERR_Import["Fehler:<br>Orchestrator<br>nicht gefunden"] --> UpdateGUIStatusErrorImp["Update GUI Status:<br>Fehler Orchestrator"] --> WFThreadEndError(Error End)
        ImportOrchestrator -- Ja --> GetExecFunc["Hole execute_full_workflow<br>Funktion"]
        GetExecFunc --> CallOrchestrator["Rufe orchestrator.execute_full_workflow<br>(prompt, params, flags, gui_status_callback)"]

        subgraph Orchestrator_Execute [Orchestrator: execute_full_workflow]
            direction TB
            OrchStart(Start) --> Step1_InputData["Schritt 1: Input-Daten (simuliert)"]
            Step1_InputData --> Call_get_input_data[rufe orchestrator.get_input_data(prompt)]
            Call_get_input_data --> Call_generate_csv["rufe gemini_perception_unit<br>.generate_prompt_based_csv(prompt)"]

            subgraph Input_Generation_Sim [Input-Generierung (Simulation): gemini_perception_unit.py]
                direction TB
                GPU_Simulate[simulate_gemini_response:<br>Nutze Templates & Keywords<br>Erzeuge Liste von Dicts]
                GPU_Parse[parse_gemini_output_to_dataframe:<br>Wandle Liste<br>in DataFrame um]
                GPU_ValidateDF[Validiere Spalten<br>des DataFrames]
                ReturnInputDF[Gebe Input DataFrame zurück]

                GPU_Simulate --> GPU_Parse
                GPU_Parse --> GPU_ValidateDF
                GPU_ValidateDF --> ReturnInputDF
            end

            Call_generate_csv --> Input_Generation_Sim
            Input_Generation_Sim --> ReceiveInputDF["Empfange Input<br>DataFrame in Orch."]
            ReceiveInputDF --> ValidateInputDF{"Input DataFrame<br>gültig (nicht None/leer)?"}
            ValidateInputDF -- Nein --> Orch_ERR_InputData["Fehler:<br>Input-Generierung<br>fehlgeschlagen"] --> OrchEndError(Error)
            ValidateInputDF -- Ja --> Step2_NeuroPersona["Schritt 2: NeuroPersona Simulation"]
            Step2_NeuroPersona --> CollectNPParams["Sammle NP Parameter<br>(epochs, lr, dr, flags...)"]
            CollectNPParams --> Call_run_neuropersona[rufe orchestrator.run_neuropersona(df, params)]
            Call_run_neuropersona --> Call_run_np_simulation["rufe neuropersona_core<br>.run_neuropersona_simulation(df, params)"]

            subgraph NeuroPersona_Core [NeuroPersona Core: run_neuropersona_simulation]
                direction TB
                NP_Start(Start) --> NP_Preprocess[preprocess_data(df)]
                NP_Preprocess --> NP_InitNodes[initialize_network_nodes(categories)]
                NP_InitNodes --> NP_CallSimulateCycle[rufe simulate_learning_cycle(...)]

                subgraph Simulation_Loop [Epochen-Schleife: simulate_learning_cycle]
                    direction TB
                    LoopInit["Init:<br>Frage-Knoten,<br>connect_network_components,<br>Historien, load_state?"]
                    LoopStart["Epoche Start<br>(tqdm loop)"]

                    SubPhase_A["A. Reset Sums<br>(node.activation_sum = 0.0)"]
                    SubPhase_B["B. Get Factors<br>(emotion, values)"]
                    SubPhase_C["C. Input<br>(activate Q_Nodes,<br>propagate_signal)"]
                    SubPhase_D["D. Propagation<br>(propagate_signal<br>for Cat/Mod/Val)"]
                    SubPhase_E["E. Activation Update<br>(sigmoid, noise,<br>history, promote)"]
                    SubPhase_F["F. Emotion Update<br>(Limbus.update_state)"]
                    SubPhase_G["G. Module Actions<br>(Creativus.ideas,<br>Criticus.evaluate,<br>Simulatrix.scenarios,<br>Socialis.update + influence)"]
                    SubPhase_H["H. Value Update<br>(ValueNode.update_value)"]
                    SubPhase_I["I. Learning & Meta-Cog<br>(MetaCog.analyze/adapt,<br>dynamic_lr, hebbian)"]
                    SubPhase_J["J. Reinforcement?<br>& Decay (decay_weights)"]
                    SubPhase_K{"K. Plasticity?<br>(STRUCTURAL_PLASTICITY_INTERVAL)"}
                    Plasticity["Prune/Sprout Connections<br>& Inactive Nodes<br>(-> all_nodes_sim kann sich ändern!)"]
                    SubPhase_L["L. Weight Logging"]
                    SubPhase_M{"M. Memory Consolidation?<br>(MEMORY_CONSOLIDATION_INTERVAL)"}
                    Consolidate["consolidate_memories<br>(nutzt PMM)"]
                    SubPhase_N["N. Get Current Nodes<br>(aus aktuellem all_nodes_sim)"]
                    SubPhase_O["O. Epoch Interpretation<br>(interpret_epoch_state)"]

                    NextEpoch{"Nächste Epoche?"}
                    EndSimLoop["Ende Epochen<br>-> Bestimme finale<br>Knotenlisten<br>aus all_nodes_sim"]

                    LoopInit --> LoopStart
                    LoopStart --> SubPhase_A --> SubPhase_B --> SubPhase_C --> SubPhase_D --> SubPhase_E --> SubPhase_F --> SubPhase_G --> SubPhase_H --> SubPhase_I --> SubPhase_J
                    SubPhase_J --> SubPhase_K
                    SubPhase_K -- Ja --> Plasticity --> SubPhase_L
                    SubPhase_K -- Nein --> SubPhase_L
                    SubPhase_L --> SubPhase_M
                    SubPhase_M -- Ja --> Consolidate --> SubPhase_N
                    SubPhase_M -- Nein --> SubPhase_N
                    SubPhase_N --> SubPhase_O
                    SubPhase_O --> NextEpoch
                    NextEpoch -- Ja --> LoopStart
                    NextEpoch -- Nein --> EndSimLoop
                end

                NP_CallSimulateCycle --> Simulation_Loop
                Simulation_Loop --> ReceiveSimResults["Empfange Historien, Logs,<br>finale Knotenlisten,<br>all_final_nodes"]
                ReceiveSimResults --> NP_GenerateReport["generate_final_report<br>(nutzt finale Knoten)"] --> ReceiveReportAndStruct["Empfange Berichtstext<br>& strukturierte Ergebnisse"]
                ReceiveReportAndStruct --> CheckPlots{"Plots<br>erstellen?"}
                CheckPlots -- Nein --> CheckSaveState
                CheckPlots -- Ja --> NP_GeneratePlots["Generiere div. Plots<br>(plot_... funktionen)"] --> CheckSaveState{"Endzustand<br>speichern?"}
                CheckSaveState -- Nein --> NP_CreateHTML
                CheckSaveState -- Ja --> NP_SaveState[save_final_network_state] --> NP_CreateHTML[create_html_report]
                NP_CreateHTML --> ReturnNPResults["Gebe Berichtstext<br>& strukturierte Ergebnisse zurück"] --> NP_End(End)

                NP_Start --> NP_Preprocess
            end

            Call_run_np_simulation --> NeuroPersona_Core
            NeuroPersona_Core --> ReceiveNPResults["Empfange NP Bericht<br>& strukturierte Ergebnisse"]
            ReceiveNPResults --> ValidateNPResults{"NP-Ergebnisse<br>gültig?"}
            ValidateNPResults -- Nein --> Orch_ERR_NPSim["Fehler:<br>NeuroPersona<br>fehlgeschlagen"] --> OrchEndError
            ValidateNPResults -- Ja --> CheckGeminiConfig{"Gemini API<br>konfiguriert?"}
            CheckGeminiConfig -- Nein --> SkipGemini["Überspringe API-Call,<br>Nutze NP-Bericht als Fallback"] --> PrepareFallbackResult["Bereite Fallback-<br>Ergebnis vor"] --> OrchEndSuccess(Result)
            CheckGeminiConfig -- Ja --> Step3_SynthesizeResponse["Schritt 3: Antwort-Synthese"]
            Step3_SynthesizeResponse --> Call_generate_final_response["rufe orchestrator.generate_final_response<br>(prompt, np_report, structured_results)"]

            subgraph Response_Synthesis_Styling [Antwort-Synthese: generate_final_response]
                direction TB
                SynthStart(Start) --> ExtractKeyResults["Extrahiere dominante Kat.,<br>Modullevel etc. aus<br>structured_results"]
                ExtractKeyResults --> BuildGeminiPrompt["Baue detaillierten<br>Gemini-Prompt zusammen<br>(Bericht, Extrakt, Anweisungen)"]
                BuildGeminiPrompt --> CallGeminiAPI[Rufe Gemini API<br>(model.generate_content)]
                CallGeminiAPI --> HandleGeminiResponse{"Antwort OK<br>& nicht blockiert?"}
                HandleGeminiResponse -- Ja --> ExtractGeminiText["Extrahiere Text<br>aus API-Antwort"] --> SynthEndSuccess(Result Text)
                HandleGeminiResponse -- Nein --> GeminiError["Fehler oder<br>Blockade bei API"] --> SynthEndError(Error Msg)
                SynthStart --> ExtractKeyResults
            end

            Call_generate_final_response --> Response_Synthesis_Styling
            Response_Synthesis_Styling -- Success --> ReceiveFinalText["Empfange finalen<br>Antworttext"] --> PrepareFinalResult["Bereite finales<br>Ergebnis vor"] --> OrchEndSuccess
            Response_Synthesis_Styling -- Error --> ReceiveFinalError["Empfange finale<br>Fehlermeldung"] --> PrepareFinalResult
        end

        CallOrchestrator --> Orchestrator_Execute
        Orchestrator_Execute -- Success --> WF_ReceiveResult["Empfange finales<br>Ergebnis (String)"] --> UpdateGUIStatusSuccess["Update GUI Status:<br>Abgeschlossen"] --> CallDisplayResult["Rufe display_final_result<br>im GUI-Thread<br>(via root.after)"]
        Orchestrator_Execute -- Error --> WF_ReceiveError["Empfange Fehler-<br>meldung (String)"] --> UpdateGUIStatusError["Update GUI Status:<br>Workflow Fehler"] --> CallDisplayResult  # Zeige auch Fehlermeldung an

        CallDisplayResult --> WFThreadEndSuccess(End)
        WFThreadEndError --> WFThreadEnd(End)
        WFThreadEndSuccess --> WFThreadEnd
    end

    %% =============================================
    %% == 3. Rückkehr zur GUI & Ergebnis         ==
    %% =============================================
    subgraph GUI_Post_Workflow
        direction TB
        DisplayResultWindow[display_final_result:<br>Zeige Toplevel-Fenster<br>mit Ergebnis/Fehler]
        UserClosesWindow{User schließt<br>Ergebnisfenster?}
        ReEnableStartButton[Reaktiviere Start-Button<br>(im finally Block)]
        GUIIdle[GUI wieder bereit]
    end

    WFThreadEnd --> ReEnableStartButton
    WFThreadEndSuccess -...-> DisplayResultWindow # Indirekter Aufruf via GUI Thread
    WFThreadEndError -...-> DisplayResultWindow # Indirekter Aufruf via GUI Thread
    DisplayResultWindow --> UserClosesWindow
    UserClosesWindow --> ReEnableStartButton
    ReEnableStartButton --> GUIIdle

    %% =============================================
    %% == 4. Verknüpfung GUI Start & Thread Ende  ==
    %% =============================================
    %% Explizite Verbindung von Thread-Start zu seinem Subgraph
    CreateWorkflowThread --> WFThreadStart

    %% Explizite Verbindung von Thread-Ende zu GUI-Aktionen
    %% WFThreadEndSuccess --> UpdateGUIStatusSuccess %% Wird bereits im Thread gemacht
    %% WFThreadEndError --> UpdateGUIStatusError %% Wird bereits im Thread gemacht
```
