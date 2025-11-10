# StarCraft II 🎮

## 🧩 Overview
A real-time strategy game where the Protoss player gathers resources, constructs buildings, and commands units to defeat a Zerg AI opponent.

## 🎮 Game Mechanics

### Core Gameplay Elements
StarCraft II is a real-time strategy game with four interconnected systems:
- **Economy**: Minerals and Vespene Gas gathered by worker units (Probes)
- **Production**: Buildings train units over time
- **Technology**: Research upgrades to enhance unit capabilities
- **Combat**: Army units engage enemy forces

### Resource System
The game tracks several resource metrics:
- **Minerals & Vespene Gas**: Consumed by actions (training units, building structures, researching upgrades)
- **Supply**: Population system with three components:
  - **Supply Cap**: Maximum population (increased by Pylons)
  - **Supply Used**: Current population consumed by units
  - **Supply Left**: Available population for new units (Cap - Used)
- **Worker Supply**: Count of Probe workers
- **Army Supply**: Population consumed by military units

### Action Execution Model
- **Multi-Action System**: Each game step requires exactly **5 actions** that execute **sequentially**
- **Resource Constraints**: Actions have costs and prerequisites:
  - Mineral/gas costs must be affordable with current resources
  - Building prerequisites must be satisfied (e.g., Gateway requires Pylon)
  - Supply must be available for unit production
- **Action Categories**:
  - **TRAIN**: Produce units from existing buildings
  - **BUILD**: Construct new buildings
  - **RESEARCH**: Unlock upgrades at specific buildings
  - **SCOUTING**: Send units to explore
  - **MULTI-ATTACK/RETREAT**: Army control commands
  - **CHRONOBOOST**: Accelerate production/research at a building
  - **EMPTY ACTION**: No operation (useful when waiting for resources)

## 🔍 Observation Space

The observation is a dictionary with three keys:
- **`obs_str`**: A string containing all game state information in text format (see example below)
- **`obs_image`**: RGB image frame from the game
- **`game_info`**: A dictionary containing followings. Note that this dictionary is constant across different runs.
  - `"player_race"`: `"Protoss"` (the player's race)
  - `"enemy_race"`: `"Zerg"` (the opponent's race)
  - `"action_dict"`: Dictionary mapping action names to action IDs (see Action Space section)
  - `"num_actions"`: `5` (number of actions to output per step)

### Observation Example (`obs_str`)

Environment provides structured summaries:

```
Summary 1: At 05:35 game time, our current situation is as follows:

Resources:
- Game time: 05:35, Worker supply: 20, Mineral: 75, Supply left: 32,
Supply cap: 54, Supply used: 22, Army supply: 1

Buildings:
count: 8
- Nexus count: 3, Pylon count: 5, Gas buildings count: 4, Warp gate

Units:
- Probe count: 20, Zealot count: 1

In Progress:

Unit producing:
- Producing probe count: 1
```

### Observation Example (`obs_image`)
![](../assets/starcraft2.png)

## 🎮 Action Space
72 discrete Protoss commands including:
- **Train units:** `TRAIN PROBE`, `TRAIN ZEALOT`, …
- **Build structures:** `BUILD PYLON`, `BUILD NEXUS`, …
- **Research upgrades:** `RESEARCH CHARGE`, `RESEARCH BLINKTECH`, …
- **Strategic:** `SCOUTING PROBE`, `MULTI-ATTACK`, `CHRONOBOOST NEXUS`, …

Following is the full list of available actions:
```
{
    'TRAIN PROBE': 0,
    'TRAIN ZEALOT': 1,
    'TRAIN ADEPT': 2,
    'TRAIN STALKER': 3,
    'TRAIN SENTRY': 4,
    'TRAIN HIGHTEMPLAR': 5,
    'TRAIN DARKTEMPLAR': 6,
    'TRAIN VOIDRAY': 7,
    'TRAIN CARRIER': 8,
    'TRAIN TEMPEST': 9,
    'TRAIN ORACLE': 10,
    'TRAIN PHOENIX': 11,
    'TRAIN MOTHERSHIP': 12,
    'TRAIN OBSERVER': 13,
    'TRAIN IMMORTAL': 14,
    'TRAIN WARPPRISM': 15,
    'TRAIN COLOSSUS': 16,
    'TRAIN DISRUPTOR': 17,
    'MORPH ARCHON': 18,
    'BUILD PYLON': 19,
    'BUILD ASSIMILATOR': 20,
    'BUILD NEXUS': 21,
    'BUILD GATEWAY': 22,
    'BUILD CYBERNETICSCORE': 23,
    'BUILD FORGE': 24,
    'BUILD TWILIGHTCOUNCIL': 25,
    'BUILD ROBOTICSFACILITY': 26,
    'BUILD STARGATE': 27,
    'BUILD TEMPLARARCHIVE': 28,
    'BUILD DARKSHRINE': 29,
    'BUILD ROBOTICSBAY': 30,
    'BUILD FLEETBEACON': 31,
    'BUILD PHOTONCANNON': 32,
    'BUILD SHIELDBATTERY': 33,
    'RESEARCH WARPGATERESEARCH': 34,
    'RESEARCH PROTOSSAIRWEAPONSLEVEL1': 35,
    'RESEARCH PROTOSSAIRWEAPONSLEVEL2': 36,
    'RESEARCH PROTOSSAIRWEAPONSLEVEL3': 37,
    'RESEARCH PROTOSSAIRARMORSLEVEL1': 38,
    'RESEARCH PROTOSSAIRARMORSLEVEL2': 39,
    'RESEARCH PROTOSSAIRARMORSLEVEL3': 40,
    'RESEARCH ADEPTPIERCINGATTACK': 41,
    'RESEARCH BLINKTECH': 42,
    'RESEARCH CHARGE': 43,
    'RESEARCH PROTOSSGROUNDWEAPONSLEVEL1': 44,
    'RESEARCH PROTOSSGROUNDWEAPONSLEVEL2': 45,
    'RESEARCH PROTOSSGROUNDWEAPONSLEVEL3': 46,
    'RESEARCH PROTOSSGROUNDARMORSLEVEL1': 47,
    'RESEARCH PROTOSSGROUNDARMORSLEVEL2': 48,
    'RESEARCH PROTOSSGROUNDARMORSLEVEL3': 49,
    'RESEARCH PROTOSSSHIELDSLEVEL1': 50,
    'RESEARCH PROTOSSSHIELDSLEVEL2': 51,
    'RESEARCH PROTOSSSHIELDSLEVEL3': 52,
    'RESEARCH EXTENDEDTHERMALLANCE': 53,
    'RESEARCH GRAVITICDRIVE': 54,
    'RESEARCH OBSERVERGRAVITICBOOSTER': 55,
    'RESEARCH PSISTORMTECH': 56,
    'RESEARCH VOIDRAYSPEEDUPGRADE': 57,
    'RESEARCH PHOENIXRANGEUPGRADE': 58,
    'RESEARCH TEMPESTGROUNDATTACKUPGRADE': 59,
    'SCOUTING PROBE': 60,
    'SCOUTING OBSERVER': 61,
    'SCOUTING ZEALOT': 62,
    'SCOUTING PHOENIX': 63,
    'MULTI-ATTACK': 64,
    'MULTI-RETREAT': 65,
    'CHRONOBOOST NEXUS': 66,
    'CHRONOBOOST CYBERNETICSCORE': 67,
    'CHRONOBOOST TWILIGHTCOUNCIL': 68,
    'CHRONOBOOST STARGATE': 69,
    'CHRONOBOOST FORGE': 70,
    'EMPTY ACTION': 71
}
```


Example output:
```
Actions
1: BUILD PYLON
2: TRAIN PROBE
3: TRAIN PROBE
4: TRAIN PROBE
5: RESEARCH CHARGE
```

## 🎯 Task Objective
Win against the built-in Zerg bot by managing economy, production, and combat.

## 🧮 Evaluation Metric
Single-player mode:
```
Score = (Wins / Total Matches) × 100
```
Multi-agent mode: Elo ratings computed from win/loss outcomes.

## ⚙️ Implementation Notes
- Default map: *Ancient Cistern LE* (Protoss vs Zerg, Hard).
- Up to 1000 steps per evaluation.
- Each response must include exactly five valid actions per step.