all: EncryptSHA512 CrackAZ99

EncryptSHA512: EncryptSHA512.c
	cc -o EncryptSHA512 EncryptSHA512.c -lcrypt

CrackAZ99: CrackAZ99.c
	cc -o CrackAZ99 CrackAZ99.c -lcrypt

clean:
	rm -f a.out *~ EncryptSHA512 CrackAZ99
