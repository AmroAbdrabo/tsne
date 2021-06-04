import numpy as np

def main():
    N = 12
    for i in range(N):
        for j in range(N):
            if j >= i:
                print(i*N+j, end="\t")
            else:
                print(end=".\t")
        print(end="\n")

    print(end="\n")



    n = 0
    while n < N:

        m = 0
        while m < n-2:
            print("[", (m  )*N + n, ",", (m+1)*N + n, ",", (m+2)*N + n, ",", (m+3)*N + n, end="] ")
            m += 4

        if m == n:
            print("[[", (n)*N + m, ",", (n)*N + m+1, ",", (n)*N + m+2, ",", (n)*N + m+3, end="] ")
            m += 4
        elif m == n-1:
            print("[", (n-1)*N + m+1, ",", (n)*N + m+1, ",", (n)*N + m+2, ",", (n)*N + m+3, end="] ")
            m += 4
        elif m == n-2:
            print("[", (n-2)*N + m+2, ",", (n-1)*N + m+2, ",", (n)*N + m+2, ",", (n)*N + m+3, end="] ")
            m += 4

        while m < N - 3:
            print("{", (n)*N + m, ",", (n)*N + m+1, ",", (n)*N + m+2, ",", (n)*N + m+3, end="] ")
            m += 4
        
        while m < N:
            print("[", (n)*N + m, end="] ")
            m += 1

        n += 1
        print(end="\n")

main()