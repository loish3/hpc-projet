#!/usr/bin/env python3
"""Partie 5 : traitement des images avec MPI (mpi4py).

Exemples :
    mpiexec -n 4 python mpi_version.py
    mpiexec -n 4 python mpi_version.py --vectorized
    mpiexec -n 4 python mpi_version.py --repeat 5
"""

import argparse
import statistics
from pathlib import Path

try:
    from mpi4py import MPI
except ImportError:
    MPI = None

import sequential_pipeline as seq
import vectorized_version as vec


def LireArguments():
    analyseur = argparse.ArgumentParser(description="Traitement des images avec MPI")
    analyseur.add_argument("--data-dir", type=Path, default=Path("dataset"))
    analyseur.add_argument("--output", type=Path, default=Path("mpi_results.csv"))
    analyseur.add_argument("--threshold", type=float, default=0.65)
    analyseur.add_argument("--vectorized", action="store_true")
    analyseur.add_argument("--repeat", type=int, default=1)
    return analyseur.parse_args()


def RepartirImages(chemins, nombre_processus):
    """Répartit les chemins entre les processus, de façon équilibrée."""
    groupes = [[] for _ in range(nombre_processus)]
    for indice, chemin in enumerate(chemins):
        groupes[indice % nombre_processus].append(chemin)
    return groupes


def TraiterImages(chemins, seuil, vectorise):
    """Traite les images attribuées à un processus."""
    if vectorise:
        traitement = vec.process_one_image_vec
    else:
        traitement = seq.process_one_image

    resultats = []
    for chemin in chemins:
        resultats.append(traitement(chemin, threshold=seuil))
    return resultats


def ExecuterSansMPI(dossier, fichier_sortie, seuil, vectorise, repetitions):
    """Mode local sans mpi4py : traite toutes les images avec un seul processus."""
    import time

    if repetitions < 1:
        print("Erreur : --repeat doit etre superieur ou egal a 1.")
        return

    chemins = sorted(dossier.glob("*.npy"))
    if not chemins:
        print(f"Aucune image .npy trouvee dans '{dossier}'.")
        return

    temps = []
    for _ in range(repetitions):
        debut = time.perf_counter()
        resultats = TraiterImages(chemins, seuil, vectorise)
        temps.append(time.perf_counter() - debut)

    resume = seq.compute_global_summary(resultats)
    seq.save_results_to_csv(resultats, output_path=fichier_sortie)
    print("=" * 68)
    print("Mode local (1 processus, sans mpi4py) : ce n'est pas une execution MPI")
    print(f"Traitement : {'vectorise' if vectorise else 'sequentiel'}")
    print(f"Nombre d'images : {len(chemins)}")
    print(f"Moyenne des moyennes : {resume['mean_of_means']:.8f}")
    print(f"Maximum global : {resume['global_max']:.8f}")
    print(f"Pixels au-dessus du seuil : {resume['total_above_threshold']:.0f}")
    print(f"Fraction moyenne au-dessus du seuil : {resume['average_fraction_above_threshold']:.8f}")
    print(f"Temps local (mediane) : {statistics.median(temps):.6f} s")
    print(f"Resultats enregistres dans : {fichier_sortie}")
    print("=" * 68)


def ExecuterMPI(dossier, fichier_sortie, seuil, vectorise, repetitions):
    if MPI is None:
        ExecuterSansMPI(dossier, fichier_sortie, seuil, vectorise, repetitions)
        return

    communication = MPI.COMM_WORLD
    rang = communication.Get_rank()
    nombre_processus = communication.Get_size()

    if repetitions < 1:
        if rang == 0:
            print("Erreur : --repeat doit etre superieur ou egal a 1.")
        return

    # Seul le processus 0 prépare la répartition.
    if rang == 0:
        chemins = sorted(dossier.glob("*.npy"))
        nombre_images = len(chemins)
        groupes = RepartirImages(chemins, nombre_processus)
    else:
        nombre_images = None
        groupes = None

    # Tous les processus reçoivent le nombre d'images.
    nombre_images = communication.bcast(nombre_images, root=0)
    if nombre_images == 0:
        if rang == 0:
            print(f"Aucune image .npy trouvee dans '{dossier}'.")
        return

    # Chaque processus reçoit uniquement sa liste de fichiers.
    mes_images = communication.scatter(groupes, root=0)
    temps = []

    for _ in range(repetitions):
        communication.Barrier()
        debut = MPI.Wtime()

        mes_resultats = TraiterImages(mes_images, seuil, vectorise)

        # Récupération des résultats sur le processus 0.
        tous_les_resultats = communication.gather(mes_resultats, root=0)
        total_images = communication.reduce(len(mes_resultats), op=MPI.SUM, root=0)

        # Le temps MPI est le temps du processus le plus lent.
        temps_local = MPI.Wtime() - debut
        temps_total = communication.reduce(temps_local, op=MPI.MAX, root=0)

        if rang == 0:
            resultats = []
            for groupe in tous_les_resultats:
                resultats.extend(groupe)
            resultats.sort(key=lambda resultat: resultat["file"])
            if total_images != nombre_images:
                raise RuntimeError("Nombre de resultats incorrect.")
            temps.append(temps_total)

    # Seul le processus 0 affiche et enregistre les résultats.
    if rang == 0:
        resume = seq.compute_global_summary(resultats)
        seq.save_results_to_csv(resultats, output_path=fichier_sortie)
        temps_median = statistics.median(temps)

        print("=" * 68)
        print(f"Version MPI : {nombre_processus} processus")
        print(f"Traitement : {'vectorise' if vectorise else 'sequentiel'}")
        print(f"Nombre d'images : {nombre_images}")
        print(f"Seuil : {seuil}")
        print(f"Repetitions : {repetitions}")
        print(f"Moyenne des moyennes : {resume['mean_of_means']:.8f}")
        print(f"Maximum global : {resume['global_max']:.8f}")
        print(f"Pixels au-dessus du seuil : {resume['total_above_threshold']:.0f}")
        print(f"Fraction moyenne au-dessus du seuil : {resume['average_fraction_above_threshold']:.8f}")
        print(f"Temps MPI (mediane) : {temps_median:.6f} s")
        print(f"Resultats enregistres dans : {fichier_sortie}")
        print("=" * 68)


def Principal():
    arguments = LireArguments()
    ExecuterMPI(arguments.data_dir, arguments.output, arguments.threshold,
                arguments.vectorized, arguments.repeat)


if __name__ == "__main__":
    Principal()
