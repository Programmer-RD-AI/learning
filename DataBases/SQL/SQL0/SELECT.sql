SELECT
    TOP (1000) [xserver_name],
    [xdttm_ins],
    [xdttm_last_ins_upd],
    [xfallback_dbid],
    [name],
    [dbid],
    [status],
    [version]
FROM
    [master].[dbo].[spt_fallback_db]
