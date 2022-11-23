
for FILE in *.yaml; do
pushd ../../..
echo "submitting config $FILE"
bash run_submit.sh --config config/adamw_runs/adamw_tuning/$FILE
popd
done