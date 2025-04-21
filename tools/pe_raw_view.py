def read_file(file_path):
    with open(file_path, 'rb') as f:
        CODE = f.read()
    CODE = CODE[CODE.find(b'\x55'):] # remove the header
    return CODE  

def print_hex_view(data, start_offset=0, bytes_per_line=16):
    offset = start_offset
    for i in range(0, len(data), bytes_per_line):
        chunk = data[i:i+bytes_per_line]
        hex_string = ' '.join(f'{b:02x}' for b in chunk)
        ascii_string = ''.join(chr(b) if 32 <= b <= 126 else '.' for b in chunk)
        Relative_offset = offset + 0x00401000
        print(f'{Relative_offset:08x}  {hex_string.ljust(bytes_per_line*3)}  {ascii_string}')
        with open("/home/fhh/ember-master/MalConv2-main/tools/hex_view_mal_6f.txt", "a") as f:
            f.write(f'{Relative_offset:08x}  {hex_string.ljust(bytes_per_line*3)}  {ascii_string}\n')
        offset += bytes_per_line

def main():
    file_path = '/home/fhh/ember-master/MalConv2-main/dataset/test/mal_6fd4849beabb6b6d40230e9f4d491d26'
    try:
        data = read_file(file_path)
        print_hex_view(data)
    except FileNotFoundError:
        print("File not found.")
    except Exception as e:
        print("An error occurred:", e)

if __name__ == "__main__":
    main()
