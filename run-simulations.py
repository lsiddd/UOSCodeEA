import sem
import argparse

ns_path = './'
script = 'UOS-LTE-v2'
campaign_dir = ns_path + 'sem'
results_dir = ns_path + 'results'

parser = argparse.ArgumentParser(description='SEM script')
parser.add_argument('-o', '--overwrite', action='store_true',
                    help='Overwrite previous campaign')
parser.add_argument('-s', '--save', action='store_true',
                    help="Don't run simulations, just save the results")
args = parser.parse_args()

campaign = sem.CampaignManager.new(ns_path, script, campaign_dir,
                                overwrite=args.overwrite, check_repo=True)
print(campaign)

param_combinations = {
    'disableDl' : 'false',
    'disableUl' : 'false',
    'enableNetAnim' : 'false',
    'enablePrediction' : ['true', 'false'],
    'epsQOS' : [500, 600],
    'epsSINR' : 600,
    'graphType' : 'false',
    'nENB' : 4,
    'nUABS' : [6, 8],
    'nUE' : [100, 200],
    'phyTraces' : 'false',
    'randomSeed' : 8005,
    'remMode' : 0,
    'scen' : 4
}

result_param = {
    'enablePrediction' : ['true', 'false'],
    'epsQOS' : [500, 600],
    'nENB' : [4],
    'nUABS' : [6, 8],
    'nUE' : [100, 200]
}

if not args.save:
    campaign.run_missing_simulations(sem.list_param_combinations(param_combinations),33)

campaign.save_to_folders(result_param, results_dir, 33)

param_combinations['nENB'] = 2
result_param['nENB'] = [2]

if not args.save:
    campaign.run_missing_simulations(sem.list_param_combinations(param_combinations),33)

campaign.save_to_folders(result_param, results_dir, 33)
