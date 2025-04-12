

**Diagramm 1: GUI & Workflow-Start**

```mermaid
graph TD
    %% =============================================
    %% == 1. GUI & Initialisierung               ==
    %% =============================================
    StartApp["User startet neuropersona_core.py"] --> CallStartGUI["rufe start_gui()"]
    %% ^-- Text in Anführungszeichen, Kommentar auf neuer Zeile
    CallStartGUI --> InitGUI["Initialisiere Tkinter GUI"]
    InitGUI --> LoadSettingsCheck{"Existiert<br>settings.json?"}
    LoadSettingsCheck -- Ja --> CallLoadGUISettings["rufe load_gui_settings()"]
    CallLoadGUISettings --> UpdateGUIWidgets["Update GUI-Widgets"]
    UpdateGUIWidgets --> GUIReady["GUI bereit"]
    LoadSettingsCheck -- Nein --> GUIReady

    GUIReady --> UserInteraction["User Interaktion"]
    UserInteraction -- Klick 'Params speichern' --> CallSaveGUISettings["rufe save_gui_settings()"]
    CallSaveGUISettings --> WriteJSON["Schreibe JSON"]
    WriteJSON --> GUIReady
    UserInteraction -- Klick 'Params laden' --> CallLoadGUISettings
    UserInteraction -- Klick 'Workflow starten' --> CallStartWorkflowAction["rufe start_full_workflow_action()"]

    CallStartWorkflowAction --> GetGUIInput["Hole GUI-Input"]
    GetGUIInput --> ValidateInput{"Input gültig?"}
    ValidateInput -- Nein --> ShowErrorMsg["Zeige Fehler"]
    ShowErrorMsg --> GUIReady
    ValidateInput -- Ja --> DisableStartButton["Deaktiviere Start-Button"]
    DisableStartButton --> CreateWorkflowThread["Erstelle & Starte Workflow Thread<br>(run_workflow_in_thread)"]
    CreateWorkflowThread --> UpdateGUIStatusInit["Update GUI Status: 'Starte...'"]
    CreateWorkflowThread --> ToDiagram2["(Siehe Diagramm 2: Workflow Thread)"]

    %% Ende dieses Diagramms
```

**Diagramm 2: Workflow Thread & Orchestrator (High-Level)**

```mermaid
graph TD
    %% =============================================
    %% == 2. Workflow Thread & Orchestrator       ==
    %% =============================================
    WFThreadStart("Workflow Thread Start<br>(run_workflow_in_thread)") --> ImportOrchestrator{"Importiere<br>orchestrator?"}
    ImportOrchestrator -- Nein --> WF_ERR_Import["Fehler"] --> UpdateGUIStatusErrorImp["Update GUI Status"] --> WFThreadEndError(Error End)
    ImportOrchestrator -- Ja --> GetExecFunc["Hole execute_full_workflow"]
    GetExecFunc --> CallOrchestrator["Rufe orchestrator.execute_full_workflow"]

    subgraph Orchestrator_Execute [Orchestrator: execute_full_workflow]
        direction TB
        OrchStart(Start) --> Step1_InputData["Schritt 1:<br>get_input_data"]
        Step1_InputData --> ValidateInputDF{"Input DF OK?"}
        ValidateInputDF -- Nein --> Orch_ERR_InputData["Error"] --> OrchEndError(Error)
        ValidateInputDF -- Ja --> Step2_NeuroPersona["Schritt 2:<br>run_neuropersona<br>(Siehe Diagramm 3)"]
        Step2_NeuroPersona --> ValidateNPResults{"NP Results OK?"}
        ValidateNPResults -- Nein --> Orch_ERR_NPSim["Error"] --> OrchEndError
        ValidateNPResults -- Ja --> CheckGeminiConfig{"Gemini API OK?"}
        CheckGeminiConfig -- Nein --> SkipGemini["Nutze NP-Bericht<br>als Fallback"] --> OrchEndSuccess(Result)
        CheckGeminiConfig -- Ja --> Step3_SynthesizeResponse["Schritt 3:<br>generate_final_response<br>(Siehe Diagramm 4)"]
        Step3_SynthesizeResponse --> SynthResult{"Synthese OK?"}
        SynthResult -- Ja --> OrchEndSuccess
        SynthResult -- Nein --> Orch_ERR_Synth["Error"] --> OrchEndError
    end

    CallOrchestrator --> Orchestrator_Execute
    Orchestrator_Execute -- Success --> WF_ReceiveResult["Empfange Resultat"] --> UpdateGUIStatusSuccess["Update GUI Status"] --> CallDisplayResult["-> display_final_result<br>(Siehe Diagramm 5)"] --> WFThreadEndSuccess(End)
    Orchestrator_Execute -- Error --> WF_ReceiveError["Empfange Fehler"] --> UpdateGUIStatusError["Update GUI Status"] --> CallDisplayResult --> WFThreadEndError

    %% Ende dieses Diagramms
```

**Diagramm 3: NeuroPersona Core Simulation (High-Level & Loop)**
*(Hier könntest du entscheiden, wie viel Detail du aus der `simulate_learning_cycle`-Schleife zeigen willst. Vielleicht nur die Hauptphasen A-O als Kette?)*

```mermaid
graph TD
    %% =============================================
    %% == 3. NeuroPersona Core Simulation         ==
    %% =============================================
    Call_run_np_simulation("Aufruf:<br>neuropersona_core.run_neuropersona_simulation") --> NP_Start(Start)
    NP_Start --> NP_Preprocess[preprocess_data]
    NP_Preprocess --> NP_InitNodes[initialize_network_nodes]
    NP_InitNodes --> NP_CallSimulateCycle[rufe simulate_learning_cycle]

    subgraph Simulation_Loop [Epochen-Schleife: simulate_learning_cycle]
        direction TB
        LoopInit["Init & Connect"] --> LoopStart["Epoche Start"]
        LoopStart --> Phases_A_to_O["Phasen A-O:<br>Reset, Input, Propagate,<br>Update Activation/Emotion/Values,<br>Modules, Learning, Decay,<br>Plasticity?, Consolidation?, Interpret"]
        Phases_A_to_O --> NextEpoch{"Nächste Epoche?"}
        NextEpoch -- Ja --> LoopStart
        NextEpoch -- Nein --> EndSimLoop["Ende Epochen"]
    end

    NP_CallSimulateCycle --> Simulation_Loop
    Simulation_Loop --> ReceiveSimResults["Empfange Ergebnisse"]
    ReceiveSimResults --> NP_GenerateReport["generate_final_report"] --> ReceiveReportAndStruct["Empfange Bericht & Struct"]
    ReceiveReportAndStruct --> CheckPlots{"Plots?"}
    CheckPlots -- Ja --> NP_GeneratePlots["Generiere Plots"] --> PostPlotActions
    CheckPlots -- Nein --> PostPlotActions
    PostPlotActions --> CheckSaveState{"Speichern?"}
    CheckSaveState -- Ja --> NP_SaveState[save_final_network_state] --> NP_CreateHTML[create_html_report]
    CheckSaveState -- Nein --> NP_CreateHTML
    NP_CreateHTML --> ReturnNPResults["Return Bericht & Struct"] --> NP_End(End)

    %% Ende dieses Diagramms
```

**Diagramm 4: Antwort-Synthese (Gemini)**

```mermaid
graph TD
    %% =============================================
    %% == 4. Antwort-Synthese (Gemini)            ==
    %% =============================================

    %% == 1. NODES DEFINITION ==
    Call_generate_final_response("Aufruf:<br>orchestrator.generate_final_response")
    InputCollectionPhase["1. Input Sammlung"]
    PromptCreationPhase["2. Prompt Erstellung"]
    GeminiProcessingPhase["3. Gemini API Verarbeitung"]
    ResultHandlingPhase["4. Ergebnis Verarbeitung"]
    ToDiagram2_Success["(Zurück zu Diagramm 2 - Erfolg)"]
    ToDiagram2_Error["(Zurück zu Diagramm 2 - Fehler)"]

    %% Nodes within Input_Sammlung Subgraph
    subgraph Input_Sammlung_Details [Details: Input Sammlung]
        direction LR
        IS_UserInput["Original User Prompt"]
        IS_NPReport["NeuroPersona Bericht<br>(Kontext)"]
        IS_NPStruct["Strukturierte NP Ergebnisse<br>(Dominanz, Modullevel...)"]
    end

    %% Nodes within Prompt_Erstellung Subgraph
    subgraph Prompt_Erstellung_Details [Details: Prompt Erstellung]
        direction TB
        PE_ExtractKeyResults["Extrahiere dominante Kat.,<br>Modullevel aus NP Struct"]
        PE_AssemblePrompt["Baue Prompt zusammen:<br>Inputs + **Instruktionen**(...)"]
    end

    %% Nodes within Gemini_Verarbeitung Subgraph
    subgraph Gemini_Verarbeitung_Details [Details: Gemini API Verarbeitung - Blackbox]
        direction TB
        GV_CallGeminiAPI["Sende Prompt an<br>Gemini API"]
        GV_InternalGeminiProcess{{"**Gemini LLM<br>(Inferenz)**<br><br><i>Verarbeitet Prompt...</i>"}}
    end

    %% Nodes within Ergebnis_Handhabung Subgraph
    subgraph Ergebnis_Handhabung_Details [Details: Ergebnis Verarbeitung]
         direction TB
         EH_HandleGeminiResponse{"Empfange Antwort<br>von API"}
         EH_CheckResponse{"Antwort OK<br>& nicht blockiert?"}
         EH_ExtractText["Extrahiere finalen<br>Antwort-Text"]
         EH_FormatError["Formatiere<br>Fehlermeldung"]
         EH_EndSuccess(Finaler Antworttext)
         EH_EndError(Fehlermeldung)
    end

    %% == 2. LINKS DEFINITION ==

    %% Hauptfluss zwischen den Phasen
    Call_generate_final_response --> InputCollectionPhase
    InputCollectionPhase --> PromptCreationPhase
    PromptCreationPhase --> GeminiProcessingPhase
    GeminiProcessingPhase --> ResultHandlingPhase

    %% Verbindungen zu und innerhalb Subgraph 1: Input Sammlung
    %% Von Hauptphase zum ersten internen Knoten
    InputCollectionPhase --> IS_UserInput
    IS_UserInput --> IS_NPReport
    IS_NPReport --> IS_NPStruct

    %% Verbindungen zu und innerhalb Subgraph 2: Prompt Erstellung
    %% Von Hauptphase zum ersten internen Knoten
    PromptCreationPhase --> PE_ExtractKeyResults
    PE_ExtractKeyResults --> PE_AssemblePrompt

    %% Verbindungen zu und innerhalb Subgraph 3: Gemini Verarbeitung
    %% Von Hauptphase zum ersten internen Knoten
    GeminiProcessingPhase --> GV_CallGeminiAPI
    GV_CallGeminiAPI --> GV_InternalGeminiProcess

    %% Verbindungen zu und innerhalb Subgraph 4: Ergebnis Handhabung
    %% Von Hauptphase zum ersten internen Knoten
    ResultHandlingPhase --> EH_HandleGeminiResponse
    EH_HandleGeminiResponse --> EH_CheckResponse
    EH_CheckResponse -- Ja --> EH_ExtractText
    EH_ExtractText --> EH_EndSuccess
    EH_CheckResponse -- Nein --> EH_FormatError
    EH_FormatError --> EH_EndError

    %% Finale Verbindungen zu den Endpunkten
    EH_EndSuccess --> ToDiagram2_Success
    EH_EndError --> ToDiagram2_Error

    %% Ende dieses Diagramms


```

**Diagramm 5: GUI Post-Workflow**

```mermaid
graph TD
    %% =============================================
    %% == 5. GUI Post-Workflow                   ==
    %% =============================================
    FromWorkflowThread("... vom Workflow Thread<br>(via root.after)") --> DisplayResultWindow["display_final_result:<br>Zeige Ergebnis-Fenster"]
    DisplayResultWindow --> UserClosesWindow{"User schließt Fenster?"}
    UserClosesWindow --> ReEnableStartButton["Reaktiviere Buttons<br>(im finally Block des Threads)"]
     ReEnableStartButton --> GUIIdle["GUI wieder bereit"]

    %% Ende dieses Diagramms
```

