# config.sh

# Account set up
ACCT=nintern

# Base path for data
basePath=/dataVault2024/dataCudaQ_QEra_July12

# Ensure the basePath exists
if [ ! -d "$basePath" ]; then
    echo "create $basePath"
    mkdir -p "$basePath"
    cd "$basePath"
    mkdir circ meas post 
    cd -
fi

# List of cx-gates and number of circuits
nCX=(100 10000 20000)
nCirc=8

# Prefix
N="mar"
