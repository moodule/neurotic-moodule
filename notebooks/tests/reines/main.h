#ifndef MAIN_H
#define MAIN_H


#include <iostream>


// CONSTANTES ////////////////////////////////////////////////////////////////////////////////

const int prob_mut = 10;	//(en %)
const int K		   = 20;


// CLASSES ///////////////////////////////////////////////////////////////////////////////////

class Genetique;

class Echiquier
{
private :
	
	int schema;
	float fit;	//qualite de l'individu en %

public :

	Echiquier ();	//l'echiquier est genere aleatoirement

	void affichage ();
	void fitness_function ();

	friend Genetique;
};


class Genetique
{
private :

	Echiquier population [K];

public :

	//Genetique ();
	
	void decoupe (int position1, int position2,  int rang);		//combine les "genomes" des 2 parents aux postions 1 et 2 de la population
	void mutation (int position);		//position de l'indiv ds la pop et non de la mutation ds le schema
	void affichage (int nb_indiv);
};

#endif
