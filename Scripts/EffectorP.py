#!/usr/bin/env python3
"""
    EffectorP: predicting fungal effector proteins from secretomes using machine learning
    Copyright (C) 2015-2016 Jana Sperschneider

    This program is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation; either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.

    Please also see the CSIRO Disclaimer provided with EffectorP (LICENCE.txt).

    Contact: jana.sperschneider@csiro.au
"""

import os
import sys
import functions
import subprocess
import errno
import uuid
import shutil
from tempfile import TemporaryDirectory


def get_weka_path(script_path):

    weka_path = os.environ.get("WEKA36")
    if weka_path is None:
        weka_path = os.path.join(script_path, 'weka-3-6-12', 'weka.jar')
    else:
        weka_path = os.path.join(weka_path, 'weka.jar')

    # -----------------------------------------------------------------------------------------------------------
    # Check that the path to the WEKA software exists
    path_exists = os.access(weka_path, os.F_OK)
    if not path_exists:
        print()
        print("Path to WEKA software does not exist!")
        print("Check the installation and the given path to the WEKA software {} in EffectorP.py (line 49).".format(weka_path))
        print("Alternatively set the environment variable 'WEKA36' to point to the folder containing the weka.jar file.")
        print()
        sys.exit(1)

    return weka_path


def get_emboss_path(script_path):
    emboss_path = shutil.which("pepstats")
    if emboss_path is None:
        emboss_path = os.path.join(script_path, 'EMBOSS-6.5.7', 'emboss')
    else:
        emboss_path = os.path.split(emboss_path)[0]

    # Check that the path to the EMBOSS software exists for pepstats
    path_exists = os.access(emboss_path, os.F_OK)
    if not path_exists:
        print()
        print("Path to EMBOSS software does not exist!")
        print("Check the installation and the given path to the EMBOSS software {} in EffectorP.py (line 70).".format(emboss_path))
        print("Alternatively, make sure that pepstats is in a directory on your PATH.")
        print()
        sys.exit(1)

    return emboss_path


def run_pepstats(RESULTS_PATH, FOLDER_IDENTIFIER, SEQUENCES, PEPSTATS_PATH, ORIGINAL_IDENTIFIERS):
    # Extract the identifiers and sequences from input FASTA file
    # Write new FASTA file with short identifiers because pepstats can't handle long names

    f_output = os.path.join(RESULTS_PATH, FOLDER_IDENTIFIER + '_short_ids.fasta')
    SHORT_IDENTIFIERS = functions.write_FASTA_short_ids(f_output, ORIGINAL_IDENTIFIERS, SEQUENCES)

    # Call pepstats
    print('Call pepstats...')

    ProcessExe = os.path.join(PEPSTATS_PATH, 'pepstats')
    ParamList = [
        ProcessExe, '-sequence',
        os.path.join(RESULTS_PATH, FOLDER_IDENTIFIER + '_short_ids.fasta'),
        '-outfile', os.path.join(RESULTS_PATH, FOLDER_IDENTIFIER + '.pepstats')
    ]

    try:
        Process = subprocess.Popen(ParamList, shell=False)
        sts = Process.wait()
        cstdout, cstderr = Process.communicate()

        if Process.returncode:
            raise Exception("Calling pepstats returned %s" % Process.returncode)
        if cstdout:
            pass
        elif cstderr:
            sys.exit()
    except:
        e = sys.exc_info()[1]
        print("Error calling pepstats: %s" % e)
        sys.exit(1)

    print('Done.\n')

    # Parse pepstats file
    print('Scan pepstats file')
    pepstats_dic = functions.pepstats(
        SHORT_IDENTIFIERS, SEQUENCES,
        os.path.join(RESULTS_PATH, FOLDER_IDENTIFIER + '.pepstats')
    )

    print('Done.\n')
    return pepstats_dic, SHORT_IDENTIFIERS


def run_weka(RESULTS_PATH, FOLDER_IDENTIFIER, pepstats_dic, SCRIPT_PATH, WEKA_PATH, ORIGINAL_IDENTIFIERS, SEQUENCES, SHORT_IDENTIFIERS):
    # Write the WEKA arff file for classification of the input FASTA file
    weka_input = os.path.join(RESULTS_PATH, FOLDER_IDENTIFIER + '.arff')
    functions.write_weka_input(weka_input, SHORT_IDENTIFIERS, pepstats_dic)

    # Call WEKA Naive Bayes model for classification of input FASTA file
    print('Start classification with EffectorP...')

    ParamList = [
        'java', '-cp', WEKA_PATH, 'weka.classifiers.bayes.NaiveBayes',
        '-l', os.path.join(SCRIPT_PATH, 'trainingdata_samegenomes_iteration15_ratio3_bayes.model'),
        '-T', os.path.join(RESULTS_PATH, FOLDER_IDENTIFIER + '.arff'),
        '-p', 'first-last'
    ]

    with open(os.path.join(RESULTS_PATH, FOLDER_IDENTIFIER + '_Predictions.txt'), 'wb') as out:
        try:
            Process = subprocess.Popen(ParamList, shell=False, stdout=out)
            sts = Process.wait()
            cstdout, cstderr = Process.communicate()

            if Process.returncode:
                raise Exception("Calling WEKA returned %s" % Process.returncode)
            if cstdout:
                pass
            elif cstderr:
                sys.exit(1)
        except:
            e = sys.exc_info()[1]
            print("Error calling WEKA: %s" % e)
            sys.exit(1)
        print('Done.\n')
        print('-----------------')

    # Parse the WEKA output file
    file_input = os.path.join(RESULTS_PATH, FOLDER_IDENTIFIER + '_Predictions.txt')
    predicted_effectors, predictions = functions.parse_weka_output(file_input, ORIGINAL_IDENTIFIERS, SEQUENCES)
    return predicted_effectors, predictions


def main():
    SCRIPT_PATH = sys.path[0]
    # Change the path to WEKA to the appropriate location on your computer
    WEKA_PATH = get_weka_path(SCRIPT_PATH)
    PEPSTATS_PATH = get_emboss_path(SCRIPT_PATH)

    commandline = sys.argv[1:]

    if commandline:
        FASTA_FILE, short_format, output_file, effector_output = functions.scan_arguments(commandline)
	# If no FASTA file was provided with the -i option
        if not FASTA_FILE:
            print()
            print('Please specify a FASTA input file using the -i option!')
            functions.usage()
    else:
        functions.usage()

    # Check if FASTA file exists
    try:
        open(FASTA_FILE, 'r')
    except OSError as e:
        print("Unable to open FASTA file:", FASTA_FILE)  # Does not exist OR no read permissions
        print("I/O error({0}): {1}".format(e.errno, e.strerror))
        sys.exit(1)

    ORIGINAL_IDENTIFIERS, SEQUENCES = functions.get_seqs_ids_fasta(FASTA_FILE)

    print('-----------------')
    print()
    print("EffectorP is running for", len(ORIGINAL_IDENTIFIERS), "proteins given in FASTA file", FASTA_FILE)
    print()

    # Temporary folder name identifier that will be used to store results
    FOLDER_IDENTIFIER = str(uuid.uuid4())

    # Path to temporary results folder
    with TemporaryDirectory() as RESULTS_PATH:
        pepstats_dic, SHORT_IDENTIFIERS = run_pepstats(RESULTS_PATH, FOLDER_IDENTIFIER, SEQUENCES, PEPSTATS_PATH, ORIGINAL_IDENTIFIERS)
        predicted_effectors, predictions = run_weka(RESULTS_PATH, FOLDER_IDENTIFIER, pepstats_dic, SCRIPT_PATH, WEKA_PATH, ORIGINAL_IDENTIFIERS, SEQUENCES, SHORT_IDENTIFIERS)

    # If user wants the stdout output directed to a specified file
    if output_file:

        with open(output_file, 'w') as out:
            # Short format: output predictions for all proteins as tab-delimited table
            if short_format:
                out.writelines(functions.short_output(predictions))
            # If the user wants to see the long format, output additional information and stats
            else:
                out.writelines(functions.short_output(predictions))
                out.writelines(functions.long_output(ORIGINAL_IDENTIFIERS, predicted_effectors))
        print('EffectorP results were saved to output file:', output_file)

    else:
        # Short format: output predictions for all proteins as tab-delimited table to stdout
        if short_format:
            print(functions.short_output(predictions))
        # If the user wants to see the long format, output additional information and stats
        else:
            print(functions.short_output(predictions))
            print(functions.long_output(ORIGINAL_IDENTIFIERS, predicted_effectors))

    # If the user additionally wants to save the predicted effectors in a provided FASTA file
    if effector_output:
        with open(effector_output, 'w') as f_output:
            for effector, prob, sequence in predicted_effectors:
                f_output.writelines('>' + effector + ' | Effector probability: ' + str(prob) + '\n')
                f_output.writelines(sequence + '\n')
    return

if __name__ == '__main__':
    main()
