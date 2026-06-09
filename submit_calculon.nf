nextflow.enable.dsl = 2

/*
 * Parameters
 */
params.config_file = "${workflow.launchDir}/config_euclidtr1v2p0multi.yaml"
params.dir_out = "${workflow.launchDir}/euclidtr1v2p0multi"


process euclid_proj_grid {

    tag "grid task ${task_id}"

    executor 'slurm'
    cpus 8
    memory '15600 MB'
    time '12:00:00'
    queue 'cpu-daily'
    clusterOptions '--clusters=calc-cpu'
    errorStrategy 'retry'
    maxRetries 1
    array 16

    input:
    val task_id

    output:
    path "done_grid_${task_id}.txt"

    script:
    """
    source "${workflow.launchDir}/activate"
    pixi run uv run python -m cosmogridv11.apps.run_probemaps \\
        --config=${params.config_file} \\
        --dir_out=${params.dir_out} \\
        --num_maps_per_index=10 \\
        --tasks=${task_id} \\
        --verbosity=info

    touch done_grid_${task_id}.txt
    """
    //    uv run python -m cosmogridv11.apps.run_probemaps --config=config_euclidtr1v2p0multi.yaml --dir_out=euclidtr1v2p0multi --num_maps_per_index=10 --tasks=0 --verbosity=info
}


process euclid_proj_fidu {

    tag "fidu task ${task_id}"

    executor 'slurm'
    cpus 8
    memory '15600 MB'
    time '12:00:00'
    queue 'cpu-daily'
    clusterOptions '--clusters=calc-cpu'
    errorStrategy 'retry'
    maxRetries 1
    array 10

    input:
    val task_id

    output:
    path "done_fidu_${task_id}.txt"

    script:
    """
    source "${workflow.launchDir}/activate"
    pixi run uv run python -m cosmogridv11.apps.run_probemaps \\
        --config=${params.config_file} \\
        --dir_out=${params.dir_out} \\
        --num_maps_per_index=10 \\
        --tasks=${task_id} \\
        --verbosity=info

    touch done_fidu_${task_id}.txt
    """
}


workflow {

    grid_tasks = Channel.from(17..32)
    fidu_tasks = Channel.from(0..9)

    euclid_proj_grid(grid_tasks)
    // euclid_proj_fidu(fidu_tasks)
}