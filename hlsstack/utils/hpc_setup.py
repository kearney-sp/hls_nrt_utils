from dask.distributed import LocalCluster, Client
import dask
from importlib.metadata import version
import os


def launch_dask(cluster_loc='local',
                hls=False,
                aws=False,
                num_processes=1,
                num_threads_per_processes=2, 
                mem_gb_per=2.5,
                num_jobs=16,
                partition='scavenger', 
                slurm_opts={'interface': 'ens7f0'},
                extra_directives=[],
                worker_args=["--lifetime", "2h", "--lifetime-stagger", "4m"],
                duration='02:00:00',
                wait_for_workers=False,
                wait_proportion=0.5,
                wait_timeout=120,
                use_nanny=True,
                debug=False):
    if cluster_loc == 'local':
        print('   setting up Local cluster...')
        import dask
        dask.config.set({'distributed.worker.daemon': False})
        if hls:
            from hlsstack.hls_funcs import fetch
            fetch.setup_env(aws=aws)
        cluster = LocalCluster(n_workers=num_processes*num_jobs,
                               threads_per_worker=num_threads_per_processes)
        client = Client(cluster)
        display(client)
    elif cluster_loc == 'hpc':
        import dask_jobqueue as jq
        djq_version = float('.'.join(version('dask_jobqueue').split('.')[:-1]))
        print('   setting up cluster on HPC...')
        if hls:
            from hlsstack.hls_funcs import fetch
            fetch.setup_env(aws=aws)
        if debug:
            import logging
            logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)
            if not os.path.exists('debug/'):
                os.mkdir('debug/')
            output_cmd = ["--output=debug/slurm-%j.out",
                          "--error=debug/slurm-%j.err"]
        else:
            output_cmd = ["--output=/dev/null",
                          "--error=/dev/null"]
        mem = mem_gb_per*num_processes*num_threads_per_processes
        n_cores_per_job = num_processes*num_threads_per_processes
        if djq_version >= 0.8:
            clust = jq.SLURMCluster(queue=partition,
                                    processes=num_processes,
                                    cores=n_cores_per_job,
                                    memory=str(mem)+'GB',
                                    #interface='ib0',
                                    #interface='ens7f0',
                                    scheduler_options=slurm_opts,
                                    local_directory='$TMPDIR',
                                    death_timeout=wait_timeout,
                                    walltime=duration,
                                    nanny=use_nanny,
                                    job_extra_directives=["--nodes=1"] + output_cmd + extra_directives,
                                    worker_extra_args=worker_args
                                   )
        else:
            clust = jq.SLURMCluster(queue=partition,
                                    processes=num_processes,
                                    cores=n_cores_per_job,
                                    memory=str(mem)+'GB',
                                    #interface='ib0',
                                    #interface='ens7f0',
                                    scheduler_options=slurm_opts,
                                    local_directory='$TMPDIR',
                                    death_timeout=wait_timeout,
                                    walltime=duration,
                                    nanny=use_nanny,
                                    job_extra=["--nodes=1"] + output_cmd + extra_directives,
                                    extra=worker_args
                                    )
            
        client=Client(clust)
        #Scale Cluster 
        #clust.scale(jobs=num_jobs)
        clust.adapt(minimum=0, maximum=num_jobs*2)
        if wait_for_workers:
            try:
                client.wait_for_workers(n_workers=int(num_jobs*num_processes*wait_proportion), timeout=wait_timeout)
            except dask.distributed.TimeoutError as e:
                print(str(num_jobs*num_processes) + ' workers may not be available. Displaying available workers.')
                #print(e)
                pass
    return client