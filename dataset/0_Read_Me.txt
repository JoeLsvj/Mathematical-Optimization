Dataset linked to the article:
The road train optimization problem with load assignment
(Le problème d'optimisation du chargement des trains routiers)

Authors:
Eliseu Araujo (araujo.eliseu28@gmail)
Maryam Darvish (maryam.darvish@fsa.ulaval.ca,https://www4.fsa.ulaval.ca/enseignant/maryam-darvish/)
Jacques Renaud (jacques.renaud@fsa.ulaval.ca, https://www4.fsa.ulaval.ca/enseignant/jacques-renaud/)

The dataset contains 160 problems.
The files are regular text files (in .txt format).
The files are named as follows:
    TL[4,8,12,16]_LTL[4,8,12,16,20]_[L,H]_[A,B,C].

[4,8,12,16] represents the number of full load orders (TL - truck load)
[4,8,12,16,20] represents the number of less-than-truckload orders (LTL)
[L,H] represents the size of the LTL load orders (L for small, H for large)
[A, B, C] an additional instance identifier.

So TL4_LTL8_L_C.txt is th problem with 4 TL orders, 8 LTL orders; orders are small-size, 
and this problem is the C one of the same configuration.


The files have the following structure:

D T Q F U M
1 10 12 24
2 18 38 24
3 36 27 24
4 36 40 24
5 37 26 8
6 29 36 9
7 32 37 8
8 4 47 9
9 49 3
10 22 13
11 15 35

First line:
D: Number of customers (thus lines 2 to D+1)
T: Number of terminals (therefore lines D+2 to D+T+1)
Q: Capacity of a trailer
F: Fixed cost of a vehicle
U: Cost per unit transported
M: Number of road trains (drivers)

Lines 2 to D+1:
Customer number; X coordinate; Y coordinate; quantity requested

Lines D+2 to D+T+1:
Terminal number; X coordinate; Y coordinate


