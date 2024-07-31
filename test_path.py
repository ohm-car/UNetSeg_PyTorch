from pathlib import Path
import os

x = Path().resolve()
print(x)
print(os.getcwd())
print(Path(__file__).resolve())
print(Path(__file__).resolve().parent)