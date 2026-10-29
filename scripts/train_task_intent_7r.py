#!/usr/bin/env python3

import hashlib
import json
import random
import re
import unicodedata
from pathlib import Path

from package_paths import CONFIG_DIR

import torch
from torch import nn


SEED = 42
FEATURE_DIM = 1024

LABELS = [
    "DRAW_CIRCLE",
    "DRAW_LINE",
    "DRAW_SQUARE",
    "STOP",
    "UNKNOWN",
]

TRAINING_DATA = {
    "DRAW_CIRCLE": [
        "円を描いて",
        "円を書いて",
        "丸を描いて",
        "丸を書いて",
        "円形の軌跡を描いて",
        "円運動をして",
        "丸い軌跡を動いて",
        "ペンで円を描いて",
        "半径25ミリの円を描いて",
        "姿勢を保って円を描いて",
        "円周をなぞって",
        "丸く動かして",
        "draw a circle",
        "draw a circular path",
        "trace a circle",
        "trace a circular path",
        "move in a circle",
        "make a round trajectory",
    ],
    "DRAW_LINE": [
        "直線を描いて",
        "線を引いて",
        "まっすぐ動いて",
        "直線軌道を描いて",
        "ペンで線を書いて",
        "横線を描いて",
        "縦線を描いて",
        "draw a line",
        "draw a straight line",
        "move in a straight line",
        "trace a line",
    ],
    "DRAW_SQUARE": [
        "四角を描いて",
        "正方形を描いて",
        "矩形を描いて",
        "四角形を書いて",
        "箱形の軌跡を描いて",
        "draw a square",
        "draw a rectangle",
        "trace a square path",
        "make a box trajectory",
    ],
    "STOP": [
        "停止して",
        "止まって",
        "動作を中止して",
        "緊急停止",
        "実行を止めて",
        "stop",
        "stop motion",
        "cancel the task",
        "halt",
        "abort",
    ],
    "UNKNOWN": [
        "こんにちは",
        "現在の状態を教えて",
        "ロボットを見せて",
        "何ができますか",
        "速度を確認して",
        "軸受を表示して",
        "hello",
        "show robot status",
        "what can you do",
        "check the controller",
    ],
}


def normalize(text):
    text = unicodedata.normalize("NFKC", text)
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


def grams(text):
    text = normalize(text)
    padded = "^" + text + "$"

    values = []

    for size in (1, 2, 3):
        for index in range(
            len(padded) - size + 1
        ):
            values.append(
                padded[index:index + size]
            )

    return values


def feature_index(token):
    digest = hashlib.sha256(
        token.encode("utf-8")
    ).digest()

    return int.from_bytes(
        digest[:8],
        "little",
    ) % FEATURE_DIM


def vectorize(text):
    vector = torch.zeros(
        FEATURE_DIM,
        dtype=torch.float32,
    )

    tokens = grams(text)

    for token in tokens:
        vector[feature_index(token)] += 1.0

    norm = torch.linalg.vector_norm(vector)

    if norm > 0:
        vector /= norm

    return vector


class IntentClassifier(nn.Module):
    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(FEATURE_DIM, 128),
            nn.ReLU(),
            nn.Dropout(0.10),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, len(LABELS)),
        )

    def forward(self, value):
        return self.network(value)


def build_dataset():
    rows = []

    for label, phrases in TRAINING_DATA.items():
        label_index = LABELS.index(label)

        for phrase in phrases:
            rows.append(
                (
                    vectorize(phrase),
                    label_index,
                    phrase,
                )
            )

    return rows


def predict(model, text):
    model.eval()

    with torch.no_grad():
        logits = model(
            vectorize(text).unsqueeze(0)
        )[0]

        probabilities = torch.softmax(
            logits,
            dim=0,
        )

        confidence, index = torch.max(
            probabilities,
            dim=0,
        )

    return (
        LABELS[int(index)],
        float(confidence),
        {
            label: float(probabilities[i])
            for i, label in enumerate(LABELS)
        },
    )


def main():
    random.seed(SEED)
    torch.manual_seed(SEED)

    rows = build_dataset()

    features = torch.stack(
        [row[0] for row in rows]
    )

    targets = torch.tensor(
        [row[1] for row in rows],
        dtype=torch.long,
    )

    model = IntentClassifier()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.004,
        weight_decay=1e-5,
    )

    criterion = nn.CrossEntropyLoss()

    model.train()

    for epoch in range(800):
        optimizer.zero_grad()

        logits = model(features)
        loss = criterion(logits, targets)

        loss.backward()
        optimizer.step()

    model.eval()

    with torch.no_grad():
        predicted = model(features).argmax(dim=1)

        training_accuracy = float(
            (predicted == targets)
            .float()
            .mean()
        )

    output = CONFIG_DIR / "task_intent_7r.pt"

    torch.save(
        {
            "state_dict": model.state_dict(),
            "labels": LABELS,
            "feature_dim": FEATURE_DIM,
            "seed": SEED,
            "model_type": (
                "hashed_character_ngram_mlp"
            ),
            "training_example_count": len(rows),
        },
        output,
    )

    print(
        "model_type="
        "hashed_character_ngram_mlp"
    )
    print(
        "training_example_count="
        + str(len(rows))
    )
    print(
        "training_accuracy="
        + format(training_accuracy, ".6f")
    )
    print("model_file=" + str(output))

    tests = [
        "円を描いて",
        "ペンで丸い線を書いて",
        "円形に動作してください",
        "draw a circular trajectory",
        "直線を描いて",
        "四角を書いて",
        "動作を止めて",
        "軸受を見せて",
    ]

    print()
    print("===== INFERENCE TEST =====")

    for text in tests:
        label, confidence, probabilities = predict(
            model,
            text,
        )

        print(
            json.dumps(
                {
                    "input": text,
                    "intent": label,
                    "confidence": round(
                        confidence,
                        6,
                    ),
                    "probabilities": {
                        key: round(value, 6)
                        for key, value
                        in probabilities.items()
                    },
                },
                ensure_ascii=False,
            )
        )

    required_tests = {
        "円を描いて": "DRAW_CIRCLE",
        "draw a circular trajectory": (
            "DRAW_CIRCLE"
        ),
        "直線を描いて": "DRAW_LINE",
        "四角を書いて": "DRAW_SQUARE",
        "動作を止めて": "STOP",
    }

    for text, expected in required_tests.items():
        actual, confidence, _ = predict(
            model,
            text,
        )

        if actual != expected:
            raise RuntimeError(
                text
                + ": expected "
                + expected
                + ", got "
                + actual
            )

        print(
            "required_test=PASS"
            + " input="
            + text
            + " intent="
            + actual
            + " confidence="
            + format(confidence, ".6f")
        )

    print()
    print("7R_AI_INTENT_MODEL=PASS")


if __name__ == "__main__":
    main()
