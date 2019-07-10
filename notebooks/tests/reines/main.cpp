
#include <iostream>
#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <math.h>

#include "main.h"


// METHODES DE LA CLASSE ECHIQUIER /////////////////////////////////////////////////////////////

Echiquier::Echiquier ()
{
	int i;
	
	schema = 0;

	for (i=0 ; i<8 ; i++)
		schema = schema + ((rand () % 7) + 1) * ((int) pow (10., i));
}


void Echiquier::affichage ()
{
	int position, temp, i, j;

	temp = schema;

	//cout << " --------\n";
	printf (" --------\n");

	for (i=1 ; i<9 ; i++)
	{
		position = temp % ((int) pow(10., i));		//pb : il y a un facteur 10^(i-1)
		temp = temp - position;
		position = (int) position / ((int) pow(10., i-1));
		
		//cout << "|";
		printf ("|");

		for (j=1 ; j<position ; j++)
			printf ("0");

		printf ("1");

		for (j=position ; j<8 ; j++)
			printf ("0");

		printf ("|\n");
	}

	printf (" --------\n\n\n\n");
}


void Echiquier::fitness_function ()
{
	int res, position1, position2, temp1, temp2, i, j;
	
	temp1 = schema;
	res = 0;

	for (i=1 ; i<9 ; i++)
	{
		position1 =  temp1 % ((int) pow(10., i));
		temp1 = temp1 - position1;
		position1 = (int) position1 / ((int) pow(10., i-1));
		temp2 = schema;

		for (j=1 ; j<9 ; j++)
		{
			position2 =  temp2 % ((int) pow(10., i));
			temp2 = temp2 - position1;
			position2 = (int) position1 / ((int) pow(10., i-1));

			if (j != i)		//les if internes s'entre-exluent bien sûr
			{
				if (position1 == position2)		// les reines considerees sont sur la meme colonne
					res++;

				if ((j-i) == (position1 - position2))		// sur la meme anti-diagonale (vu la façon dt l'echiquier est affiché)
					res++;

				if ((j-i) == (position2 - position1))		// sur la meme diagonale
					res++;
			}
		}
	}

	res = 56 - res;
	fit = (((float) res) / 56.) * 100. ;
}



// METHODES DE LA CLASSE GENETIQUE ///////////////////////////////////////////////////////////

void Genetique::decoupe (int position1, int position2, int rang)
{
	int temp1, temp2;

	temp1 = population [position1].schema % (((int) pow(10., rang)) + population [position2].schema - (population [position2].schema % ((int) pow(10., rang))));
	temp2 = population [position2].schema % (((int) pow(10., rang)) + population [position1].schema - (population [position1].schema % ((int) pow(10., rang))));

	population [position1].schema = temp1;
	population [position2].schema = temp2;
}


void Genetique::mutation (int position)
{
	int proba, rang, nv_pos, temp1, temp2;

	proba = rand () % 100;

	if (proba <= prob_mut)
	{
		rang = (rand () % 7) + 1;
		nv_pos = (rand () % 7) + 1;
		temp1 = population [position].schema - (population [position].schema % ((int) pow(10., rang)));
		temp2 = population [position].schema % ((int) pow(10., rang-1)) + nv_pos * ((int) pow(10., rang-1));

		population [position].schema = temp1 + temp2;
	}
}


void Genetique::affichage (int nb_indiv)
{
	int i;
	
	for (i=0 ; i<nb_indiv ; i++)
		population [i].affichage ();
}


// FONCTIONS GLOBALES ////////////////////////////////////////////////////////////////////////

int main ()
{
	Genetique test;

	test.affichage (2);

	return 0;
}
