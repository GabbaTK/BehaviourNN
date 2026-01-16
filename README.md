# BehaviourNN

BehaviourNN is a small experimental game that trains a neural network to model how you make choices, and then quietly uses that model against you.

# Concept

You are repeatedly presented with simple binary choices, usually between two unrelated words.

There are no instructions on how to choose.
There is no obvious meaning to the words.
There is no visible strategy.

You are asked only to choose as you normally would.

Over time, the game records:
 - what you chose
 - what alternatives you were shown
 - how consistently you decide

After enough sessions, a neural network is trained to predict your future choices probabilistically, not deterministically.
Later, the game introduces consequences based on how confident the model is about you.
You are, in effect, playing against a model of yourself.

# Structure

The project consists of two parts:

1. Client (Game)

- Console-based
- Presents choices
- Enforces time delays between sessions
- Records decisions locally
- Uploads data to the server
- Queries the trained model once it exists

2. Server

- Receives choice histories
- Trains a per-player neural network asynchronously
- Stores models by GUID

Responds to:
- Training status queries
- Inference requests (probabilities)

Each player gets their own model.

Gameplay Rules (Hard-coded by design)
These rules are intentional and not configurable by the player:

- You can only play once every 2 hours
- Each session contains 30 choices
- A minimum of 5 sessions (150 choices) is required before the model is allowed to influence the game
- The model is trained with label smoothing and dropout

The model outputs probabilities, not decisions
The model learns how likely you are to choose the left option in a given pairing
If your behavior is inconsistent, the model will reflect that.
If you change over time, the model will lag.
If you try to “outsmart” it, the uncertainty increases.

# Running the Project
## Client
### Requirements
- Python 3.14

### Running
```python
python game.py <server_ip>
```
If the server IP is not provided, it defaults to connecting to the master server

### Optional commands
```
--status           # Check training status
--resend           # Re-upload saved data
--inference L R    # Query model probabilities manually
```

## Server
### Requirements
- Python 3.14
- Pytorch

### Running
```python
python server.py
```

# Data & Privacy
Choices are stored locally in save.dat

Each player is identified by a generated GUID which contains their username + a random UUID

The server stores:
- The model
- Training status

# Notes
The model will plateau, this is expected

Loss will not approach zero, this is expected

Uncertainty is a feature, not a bug

# License

This project is experimental and provided as-is.

Use, modify, and study it freely.