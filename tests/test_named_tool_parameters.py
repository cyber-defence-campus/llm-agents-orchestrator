"""Recover the complete flat argument syntax observed in native audit chats."""
from agent_framework.llm.utils import parse_tool_invocations


def test_named_xml_children_preserve_explicit_values_in_function_envelope():
    content='''<function=audit_publish>
<recipient>integration</recipient>
<refs>["E1#/revision_observations/0/revision"]</refs>
<receipts>["receipt-1"]</receipts>
<question>Revision &amp; ownership evidence</question>
</function>'''
    result=parse_tool_invocations(content)
    assert result==[{'toolName':'audit_publish','args':{
        'recipient':'integration','refs':'["E1#/revision_observations/0/revision"]',
        'receipts':'["receipt-1"]','question':'Revision & ownership evidence'}}]


def test_named_recovery_does_not_convert_prose_duplicates_or_mixed_dialects():
    for body in ('Please send <recipient>integration</recipient>',
                 '<recipient>integration</recipient><recipient>events</recipient>',
                 '<recipient>integration</recipient><parameter=refs>["E1#/x"]</parameter>'):
        result=parse_tool_invocations('<function=audit_publish>'+body+'</function>')
        assert 'recipient' not in result[0]['args']
    assert parse_tool_invocations('<recipient>integration</recipient>') is None
