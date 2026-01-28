from config import load_secrets, get_env

def main():
    load_secrets()
    user = get_env("FREELANCEMAP_USERNAME")
    print("Loaded username length:", len(user))  # avoids printing the secret

if __name__ == "__main__":
    main()
