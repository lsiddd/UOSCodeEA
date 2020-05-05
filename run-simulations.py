import sem
import argparse

ns_path = './'
script = 'UOS-LTE-v2'
campaign_dir = ns_path + 'sem'
results_dir = ns_path + 'results'

parser = argparse.ArgumentParser(description='SEM script')
parser.add_argument('-o', '--overwrite', action='store_true',
                    help='Overwrite previous campaign')
args = parser.parse_args()

campaign = sem.CampaignManager.new(ns_path, script, campaign_dir, overwrite=args.overwrite)
print(campaign)

param_combinations = {
    'disableDl' : 'false',
    'disableUl' : 'false',
    'enableNetAnim' : 'false',
    'graphType' : 'false',
    'nENB' : [2, 4],
    'nUABS' : 6,
    'nUE' : [100, 200],
    'randomSeed' : 8000,
    'remMode' : 0,
    'scen' : [3, 4]
}

campaign.run_missing_simulations(sem.list_param_combinations(param_combinations),33)

result_param = { 
    'nENB' : [2, 4], 
    'nUABS' : [6],
    'nUE' : [100, 200],
    'scen' : [3, 4]
}

campaign.save_to_folders(result_param, results_dir, 33)
