# config.sh

# Account set up
ACCT=nintern

# List of cx-gates and number of circuits
nCX=(100)

# this setup only affect the prep_gateList
nCirc=8

# Shots of sampling
#shots=(10000 1000000 10000000 100000000)
shots=(1)
# Prefix
# run gate list has a hardcode mar
N="mar"

qft=1

# nvidia options
option="fp32,mgpu"