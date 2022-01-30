Authors: Antonio Marino, Zach Glaser, and Hugh Shanno

Project: Christofides Serdyukov Algorithm Implementation

This Zip file contains the following:

README.txt: this readme
Main.py: a python file that contains the algorithm
graph.txt: a small text file that contains an example graph for the algorithm to run on.

Installation Instructions:
The file can be downloaded as a zip and can be run through any python shell. In order to run the code, navigate to the appropriate directory, install the networkx package on the shell by running the commandline "pip install networkx".

File Instructions:
Any file that you wish to run the program on must be placed in the file with the main program.
The file must be formatted as follows:
All lines must contain edges, with each line containing one edge of the form: 
Vertex,Vertex,Weight
Weight must be a real number greater than 0. Vertices must not have commas or spaces in them, and there should not be spaces around the commas.

To perform the Christofides algorithm, graphs must be complete (every node must have an edge to every other node), and satisfy the triangle inequality (the shortest path between any two nodes is the edge between them).

Running Instructions:
In order to run this program, type in the commandline as follows: "python main.py graph.txt"
where graph.txt may be replaced with another text file containing a graph in the directory.