from neuralflex.inference.streaming import StreamingGenerator


if __name__ == "__main__":
    generator = StreamingGenerator()
    for token in generator.stream("Hello"):
        print(token, end="", flush=True)
