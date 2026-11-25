from agent_framework.tools.executor import _format_tool_result


def test_non_scalar_protocol_metadata_remains_generic_result():
 text,images=_format_tool_result('example',{'protocol':['application-data'],'payload':'kept'})
 assert 'kept' in text and not images


def test_typed_work_result_preserves_value_and_reuse_metadata():
 text,images=_format_tool_result('typed_work',{'state':'succeeded','value':{'accepted':True},'reused':True})
 assert 'succeeded' in text and 'accepted' in text and 'reused' in text and not images
