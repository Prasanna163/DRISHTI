param(
    [Parameter(ValueFromRemainingArguments=$true)]
    $RemainingArgs
)

python scripts\run_end_to_end.py $RemainingArgs
