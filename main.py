from app.observability import configure_logfire


def main():
    configure_logfire(service_name="kub-rag-assitant-model", service_version="0.1.0")
    print("Hello from kub-rag-assitant-model!")


if __name__ == "__main__":
    main()
