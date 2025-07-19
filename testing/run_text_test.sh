***REMOVED***

***REMOVED***
***REMOVED***
DATASET="https://gradients.s3.eu-north-1.amazonaws.com/e1230b33949f9bdf_train_data.json?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAVVZOOA7SA4UOFLPI%2F20250430%2Feu-north-1%2Fs3%2Faws4_request&X-Amz-Date=20250430T023943Z&X-Amz-Expires=604800&X-Amz-SignedHeaders=host&X-Amz-Signature=847c416b0015e19470188979543b5778ae51d1acff922d19f08487a4e6c37db9"
***REMOVED***
  "field_instruction":"question",
  "field_output":"chosen"
***REMOVED***
***REMOVED***
***REMOVED***

***REMOVED***
***REMOVED***
***REMOVED***

***REMOVED***
***REMOVED***
chmod 700 "$DATA_DIR"

***REMOVED***

***REMOVED***
***REMOVED***
***REMOVED***
  --memory=32g \
  --cpus=4 \
  --volume "$DATA_DIR:/workspace/input_data:rw" \
***REMOVED***
***REMOVED***
***REMOVED***
***REMOVED***
***REMOVED***
***REMOVED***
***REMOVED***
***REMOVED*** \
  --hours-to-complete "$HOURS_TO_COMPLETE" \
  --huggingface-token "$HUGGINGFACE_TOKEN" \
  --wandb-token "$WANDB_TOKEN" \
  --huggingface-username "$HUGGINGFACE_USERNAME"