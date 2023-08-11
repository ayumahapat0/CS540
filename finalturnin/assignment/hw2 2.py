import sys
import math
import string


def get_parameter_vectors():
    '''
    This function parses e.txt and s.txt to get the  26-dimensional multinomial
    parameter vector (characters probabilities of English and Spanish) as
    descibed in section 1.2 of the writeup

    Returns: tuple of vectors e and s
    '''
    #Implementing vectors e,s as lists (arrays) of length 26
    #with p[0] being the probability of 'A' and so on
    e=[0]*26
    s=[0]*26

    with open('e.txt',encoding='utf-8') as f:
        for line in f:
            #strip: removes the newline character
            #split: split the string on space character
            char,prob=line.strip().split(" ")
            #ord('E') gives the ASCII (integer) value of character 'E'
            #we then subtract it from 'A' to give array index
            #This way 'A' gets index 0 and 'Z' gets index 25.
            e[ord(char)-ord('A')]=float(prob)
    f.close()

    with open('s.txt',encoding='utf-8') as f:
        for line in f:
            char,prob=line.strip().split(" ")
            s[ord(char)-ord('A')]=float(prob)
    f.close()

    return (e,s)

def shred(filename):
    #Using a dictionary here. You may change this to any data structure of
    #your choice such as lists (X=[]) etc. for the assignment
    X=dict().fromkeys(string.ascii_uppercase, 0)
    with open (filename,encoding='utf-8') as f:
        for line in f:
            words = line.strip()
            for word in words:
                word.split()
                for letter in word:
                    if letter.upper() in X:
                        X[letter.upper()] += 1

    return X
def Q1():
    letters = shred("letter.txt")
    print("Q1")
    for letter in letters:
        print(letter, letters[letter])

def Q2():
    letters = shred("letter.txt")
    print("Q2")
    (english, spanish) = get_parameter_vectors()
    print("%.4f" % float(letters["A"] * math.log(english[0])))
    print("%.4f" % float(letters["A"] * math.log(spanish[0])))

def Q3():
    letters = shred("letter.txt")
    print("Q3")
    (english, spanish) = get_parameter_vectors()
    sum_english = math.log(0.6)
    sum_spanish = math.log(0.4)
    index = 0
    for letter in letters:
        sum_english += float(letters[letter.upper()] * math.log(english[index]))
        index += 1

    index = 0
    for letter in letters:
        sum_spanish += float(letters[letter.upper()] * math.log(spanish[index]))
        index += 1

    print("%.4f" % sum_english)
    print("%.4f" % sum_spanish)


def Q4():
    letters = shred("letter.txt")
    print("Q4")
    (english, spanish) = get_parameter_vectors()
    sum_english = math.log(0.6)
    sum_spanish = math.log(0.4)
    index = 0
    for letter in letters:
        sum_english += letters[letter.upper()] * math.log(english[index])
        index += 1

    index = 0
    for letter in letters:
        sum_spanish += letters[letter.upper()] * math.log(spanish[index])
        index += 1

    difference = sum_spanish - sum_english

    result = 0.0
    if difference <= -100.0:
        result = 1.0

    if -100 < difference < 100:
        result = float(1.0 / (1.0 + math.pow(math.e, difference)))

    print("%.4f" % result)


if __name__ == '__main__':
    Q1()
    Q2()
    Q3()
    Q4()

