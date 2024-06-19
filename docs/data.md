# Data

## Useful commands for processing files
### Train test split
`awk '{if(rand()<0.9) {print $0 > "output.train"} else {print $0 > "output.test"}}' input`

### Extracting json field from file
`jq -r '.text' input.json > output.txt `