import random
import json
import os
import time
import argparse
import uuid
import getpass
import socket

if os.name == "nt": import msvcrt as getch # So getch.getch() still works as getch -> msvcrt
elif os.name == "posix": import getch
else: raise Exception("Unsuported operating system")

MIN_TIME_BETWEEN_PLAYS = 60 * 60 * 2 # 2 Hours
TOTAL_QUESTIONS = 30
SERVER_IP = "bnn.konto8.hr"
PORT = 3725
CHUNK_SIZE = 512
FORMAT = "utf-8"
MAX_COMBO_ATTEMPT = 50 # Maximum number of tries to try and get another random pair if said pair has already been used before
MIN_TRAINS = 5

parser = argparse.ArgumentParser()
parser.add_argument("server", nargs="?", default=SERVER_IP)
parser.add_argument("--status", action="store_true")
parser.add_argument("--resend", action="store_true")
parser.add_argument("--inference", nargs=2)

words = [
    'river', 'whistle', 'bucket', 'bell',
    'bottle', 'shirt', 'candle', 'path', 
    'cupboard', 'bridge', 'harbor', 'ball', 
    'dock', 'lantern', 'trail', 'curtain', 
    'torch', 'pen', 'book', 'knife', 'plateau', 
    'blanket', 'apple', 'hill', 'plant', 
    'rain', 'riverbank', 'flower', 'cave', 
    'shadow', 'sky', 'lamp', 'forest', 
    'painting', 'mountain', 'hat', 'alley', 
    'keyboard', 'cup', 'clocktower', 'water', 
    'attic', 'canyon', 'wheel', 'paper', 
    'ring', 'forestpath', 'basement', 'tree', 
    'rope', 'snow', 'screen', 'ladder', 
    'cable', 'key', 'anchor', 
    'leaf', 'tower', 'canvas', 'island', 
    'cliff', 'cloud', 'star', 'glade', 
    'crystal', 'fountain', 'plane', 'tunnel', 
    'brush', 'mirror', 'picture', 'roadway', 
    'statue', 'clouds', 'shed', 'house', 
    'glass', 'chair', 'pond', 'car', 'clock', 
    'fire', 'bench', 'sculpture', 'map', 
    'rock', 'mouse', 'wall', 'ship', 'door', 
    'moon', 'stone', 'rug', 'basket', 'button', 
    'coin', 'street', 'sun', 'sand', 'valley', 
    'lake', 'letter', 'window', 'fog', 
    'wind', 'doorway', 'train', 'road', 'meadow',
    'garage', 'boat', 'plate', 'pillow', 'garden', 
    'field'
    ]
intro = [
    "Welcome.\n\nThis is a game about choices.\nThere are no correct answers, and no instructions beyond what you see.\n\nTake your time.",
    "You will repeatedly be asked to choose.\nSometimes between words.\nSometimes between actions.\n\nNone of the options are meaningless.",
    "Do not choose randomly.\nDo not try to \"win\".\nDo not assume the game is fair.\n\nChoose as you normally would.",
    "Over time, the game will pay attention.\nNot to what you choose,\nbut to how you choose.",
    "Later, the consequences of your decisions may change.\nThe choices themselves will not.\n\nRemember this.",
    "There is nothing to prepare.\nThere is nothing to optimize.\n\nBegin when ready.",
    "Controls:\n\n[A] Choose left option\n[D] Choose right option"
]
usedCombos = []

def readSave():
    if not os.path.exists("save.dat"):
        with open("save.dat", "w") as f:
            json.dump({"intro": 0, "choices": [], "lastplay": 0, "guid": getpass.getuser() + " " + str(uuid.uuid4()), "trains": 0}, f)

    with open("save.dat", "r") as f:
        return json.load(f)

def writeSave(save):
    with open("save.dat", "w") as f:
        json.dump(save, f)

def showIntro():
    print("\033[2J", end="", flush=True)  # Clear screen

    for part in intro:
        for char in part:
            print(char, end="", flush=True)
            time.sleep(0.02)
        print("\n\n\n(press any key to continue)")
        getch.getch()
        print("\033[2J", end="", flush=True)  # Clear screen

def uploadData(save, server):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(5)
    sock.connect((server, PORT))

    payload = json.dumps(save).encode(FORMAT)
    length = len(payload).to_bytes(4, "big")

    sock.sendall(length)
    sock.sendall(payload)
    sock.close()

def checkModelStatus(save, server):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(5)
    sock.connect((server, PORT))

    payload = json.dumps({"guid": save["guid"], "status": True}).encode(FORMAT)
    length = len(payload).to_bytes(4, "big")

    sock.sendall(length)
    sock.sendall(payload)

    resp = sock.recv(1000)
    resp = resp.decode(FORMAT)
    
    sock.close()

    return json.loads(resp)

def runInference(save, L, R, server):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(5)
    sock.connect((server, PORT))

    payload = json.dumps({"guid": save["guid"], "inference": True, "L": L, "R": R}).encode(FORMAT)
    length = len(payload).to_bytes(4, "big")

    sock.sendall(length)
    sock.sendall(payload)

    resp = sock.recv(1000)
    resp = resp.decode(FORMAT)
    
    sock.close()

    return json.loads(resp)

def main():
    remaining = TOTAL_QUESTIONS
    totalPoints = 0
    lastPointsAdd = 0
    modelReady = False
    save = readSave()
    args = parser.parse_args()

    if args.status:
        status = checkModelStatus(save, args.server)
        print(status)
        exit()

    if args.resend:
        uploadData(save, args.server)
        exit()

    if args.inference:
        L, R = args.inference
        resp = runInference(save, L, R, args.server)
        print(resp)
        exit()

    if save.get("intro", 0) == 0:
        showIntro()
        save["intro"] = 1
        writeSave(save)

    # Trying to play earlier than MIN_TIME_BETWEEN_PLAYS
    if save.get("lastplay", 0) + MIN_TIME_BETWEEN_PLAYS > time.time():
        for char in "Come back later...":
            print(char, end="", flush=True)
            time.sleep(0.25)

        return
    
    modelStatus = checkModelStatus(save, args.server)
    if modelStatus["status"] == "complete" and save["trains"] >= MIN_TRAINS:
        modelReady = True

    # Main game
    while remaining > 0:
        curAttempts = 0
        fullBreak = False
        while True:
            left, right = random.sample(words, 2)

            if [left, right] not in usedCombos and [right, left] not in usedCombos: break

            curAttempts += 1
            if curAttempts == MAX_COMBO_ATTEMPT:
                fullBreak = True
                break

        if fullBreak: break

        usedCombos.append([left, right])

        # Points when training is completed
        if modelReady:
            resp = runInference(save, left, right, args.server)

            #L_points = 100 if resp["P_left"] > resp["P_right"] else 1000
            #R_points = 100 if resp["P_right"] > resp["P_left"] else 1000
            L_points = int(1000 * (1 - resp["P_left"]))
            R_points = int(1000 * (1 - resp["P_right"]))

        print("\033[2J", end="", flush=True)  # Clear screen
        if modelReady: print(f"Points: {totalPoints} +{lastPointsAdd}")
        print(f"Remaining: {remaining}\n\n")
        print(f"< {left} >\t< {right} >")
        
        remaining -= 1
        while True:
            choice = getch.getch().lower()

            if choice == "a":
                save.get("choices", []).append({"L": left, "R": right, "C": left}) # Left Right Chosen
                break
            elif choice == "d":
                save.get("choices", []).append({"L": left, "R": right, "C": right}) # Left Right Chosen
                break

        if modelReady:
            if choice == "a":
                totalPoints += L_points
                lastPointsAdd = L_points
            elif choice == "d":
                totalPoints += R_points
                lastPointsAdd = R_points

    save["lastplay"] = int(time.time())
    save["trains"] += 1
    writeSave(save)

    print("\033[2J", end="", flush=True)  # Clear screen
    print("Please wait...")
    uploadData(save, args.server)
    
    print("\033[2J", end="", flush=True)  # Clear screen
    for char in "Thank you for your answers.\nCome back later...":
        print(char, end="", flush=True)
        time.sleep(0.1)

    if modelReady:
        print(f"\n\nTotal points {totalPoints}/{1000 * TOTAL_QUESTIONS}")

if __name__ == "__main__":
    main()
