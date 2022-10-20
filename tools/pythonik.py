with open('out.bin', 'wb') as outfile:
    outfile.write(bytes([139]))
    outfile.write(bytes([162]))