# Check if a packet is valid or not

packet: list[int] = [0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07]


match packet:

    case [
        c1,
        c2,
        *data,
        footer,
    ] if (  # Deconstruct packet into header, data, and footer
        checksum := c1 + c2
    ) == sum(
        data
    ) and len(  # Check that the checksum is correct
        data
    ) == footer:  # Check that the data length is correct

        print(f"Packet received: {data} (Checksum: {checksum})")

    case [
        c1,
        c2,
        *data,
    ]:  # Failure case where structure is correct but checksum is wrong

        print(f"Packet received: {data} (Checksum Failed)")

    case [_, *__]:  # Failure case where packet is too short

        print("Invalid packet length")

    case []:  # Failure case where packet is empty

        print("Empty packet")

    case _:  # Failure case where packet is invalid

        print("Invalid packet")
