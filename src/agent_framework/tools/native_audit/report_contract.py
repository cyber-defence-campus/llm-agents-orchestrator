"""Public field types; no expected answers or fixture values."""
FIELD_TYPES={
    'component':'string','check':'revision | owner | enabled',
    'classification':'consistent | violation | allowed_exception','assessment_tick':'integer',
    'proof.identity':'string source ID','proof.alias':'string source ID',
    'proof.rule':'string source ID','proof.time_rule':'string source ID',
    'proof.observed':'object with ref:string and value:original source scalar',
    'proof.observed_tick':'object with ref:string and value:integer',
    'proof.snapshot_tick':'object with ref:string and value:integer',
    'proof.baseline':'object with ref:string and value:original source scalar',
    'proof.exception':'null or object',
    'proof.exception.component':'string source ID',
    'proof.exception.owner':'object with ref:string and value:string',
    'proof.exception.from_tick':'object with ref:string and value:integer',
    'proof.exception.until_tick':'object with ref:string and value:integer',
    'proof.exception.evaluated_at':'integer','proof.exception.applies':'boolean',
}
