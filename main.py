from network import Network


def main():
    print("Hello Python!")
    mynet: Network = Network(4, [1], 2)
    mynet._get_info()


if __name__ == "__main__":
    main()
