from src import main


def test_greet():
    assert main.greet() == "Hello from TNAflix"
