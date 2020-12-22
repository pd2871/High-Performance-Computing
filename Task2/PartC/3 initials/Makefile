all: EncryptSHA512 CrackAZZ99

EncryptSHA512: EncryptSHA512.c
	cc -o EncryptSHA512 EncryptSHA512.c -lcrypt

CrackAZZ99: CrackAZZ99.c
	cc -o CrackAZZ99 CrackAZZ99.c -lcrypt

clean:
	rm -f a.out *~ EncryptSHA512 CrackAZZ99
