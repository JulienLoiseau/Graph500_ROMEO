/**
*	\file bfs.c
*	\author Julien Loiseau
*	\date 30/09/2015
*	\brief \TODO documentation
*	\TODO documentation
*/


/* Rappel sur la topologie de romeo : 
 * SWITCH 6 : 11 à 26 (16)
 * SWITCH 7 : 27 à 42 (16) (32)
 * SWITCH 8 : 43 à 58 (16) (48)
 * SWITCH 9 : 59 à 74 (16) (64)
 * SWITCH 10: 75 à 90 (16) (80)
 * SWITCH 11: 91 à 106 (16) (96)
 * SWITCH 12: 107 à 122 (16) (112)
 * SWITCH 13: 123 à 140  (18) (130)  */

#ifndef __STDC_LIMIT_MACROS
#define __STDC_LIMIT_MACROS
#endif
#ifndef __STDC_FORMAT_MACROS
#define __STDC_FORMAT_MACROS
#endif
#ifndef __STDC_CONSTANT_MACROS
#define __STDC_CONSTANT_MACROS
#endif

#include "common.hpp"
#include "generator/utils.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <omp.h>
#include <parallel/algorithm>


#include <iostream>
#include <list>


/* Directement défini dans le makefile */
//#define PROCESS_COUNT 16

#ifndef PROCESS_COUNT
#error "Define PROCESS_COUNT to yout target number of processes"
#endif

/* données process */
static int bfs_rank;
static unsigned int line_rank;
static unsigned int line_num;
static unsigned int col_rank;
static unsigned int col_num;

static int line_comm_rank;
static int col_comm_rank;

static MPI_Comm mpi_col_comm;
static MPI_Comm mpi_line_comm;

static const unsigned int constant_size = PROCESS_COUNT;
static const unsigned int lg_constant_size = log2(constant_size);
static const unsigned int line_size = sqrt(constant_size);
static const unsigned int col_size = line_size;
static const unsigned int lg_line_size = lg_constant_size / 2;


/* Static pour le moment, sera défini par calcul plus tard */
//static const uint64_t nblocks = (uint64_t)(1) << 28;
static uint32_t nblocks_line;// = (uint32_t)(1) << 16;

static inline uint32_t nvertices(const unsigned int lg_nvertices)
{
	return uint32_t(1) << lg_nvertices;
}


static inline unsigned int lg_nvertices_1d_per_proc(const unsigned int lg_nvertices)
{
	return lg_nvertices - lg_constant_size;
}

static inline uint32_t nvertices_1d_per_proc(const unsigned int lg_nvertices)
{
	return uint32_t(1) << lg_nvertices_1d_per_proc(lg_nvertices);
}

static inline unsigned int lg_nvertices_2d_per_proc(const unsigned int lg_nvertices)
{
	return lg_nvertices - lg_line_size;
}

static inline uint32_t nvertices_2d_per_proc(const unsigned int lg_nvertices)
{
	return uint32_t(1) << lg_nvertices_2d_per_proc(lg_nvertices);
}

/* TODO utile ? */
static inline unsigned int nwords_per_proc_line(const unsigned int lg_nvertices)
{
	// Ici on round up pour les cas avec peu de sommets 
	return (nvertices_2d_per_proc(lg_nvertices) + BITMAP_UNIT_SIZE  -1 )/ BITMAP_UNIT_SIZE;
}

static inline uint32_t nvertices_2d_per_block(const unsigned int lg_nvertices)
{ 
	return nvertices_2d_per_proc(lg_nvertices) / nblocks_line;
}

static inline unsigned int nwords_per_block_line(const unsigned int lg_nvertices)
{
	return (nvertices_2d_per_block(lg_nvertices) + BITMAP_UNIT_SIZE -1) / BITMAP_UNIT_SIZE;
}

static inline uint64_t nblocks(const uint32_t nblocks_line)
{
	return (uint64_t)nblocks_line*(uint64_t)nblocks_line;
}

static inline uint32_t nvertices_limit(const unsigned int lg_nvertices){
	return nvertices(lg_nvertices)*0.01;
}

inline int compute_edge_owner(uint32_t v0, uint32_t v1, unsigned int lg_nvertices)
{
	// Déterminer la colonne du sommet v0 
	int col = v0 / nvertices_2d_per_proc(lg_nvertices);
	// Déterminer la ligne du sommet v1 
	int line = v1 / nvertices_2d_per_proc(lg_nvertices);
	return line*line_size+col;
}

inline uint64_t encode_edge_uint64_t(uint32_t v0, uint32_t v1)
{
	return (((uint64_t)v0 << 32) | (uint64_t)v1);
}

inline uint32_t decode_v0_uint64_t(uint64_t edge)
{
	return (uint32_t)(edge >> 32);
}

inline uint32_t decode_v1_uint64_t(uint64_t edge)
{
	return (uint32_t)(edge);
}

inline uint32_t encode_edge_uint32_t(uint16_t v0, uint16_t v1)
{
	return (((uint32_t)v0 << 16) | (uint16_t)v1);
}




static struct make_graph_data_structure
{	

	BITMAP_UNIT restrict * in_queue;
  	BITMAP_UNIT restrict * out_queue;
  	BITMAP_UNIT restrict * my_assigned_targets;
  	BITMAP_UNIT restrict * prefix;
	BITMAP_UNIT restrict * blocks;

	int32_t restrict * visited_v2;
	uint32_t   n_in_queue;
 	uint32_t n_out_queue;
  	
	unsigned int * CSR_C;
	unsigned int * CSR_R;

	unsigned int * CSC_R;
	unsigned int * CSC_C;

	unsigned int lg_nvertices; /* Global count */	
	unsigned int rank_in_row;

	MPI_Comm comm_per_row;  	
  	
} graph;

/*-----------  Fonction obligatoires pour la validation ---------------*/
void get_vertex_distribution_for_pred(size_t count, const int64_t* vertex_p, int* owner_p, size_t* local_p) {
  const int64_t* restrict vertex = vertex_p;
  int* restrict owner = owner_p;
  size_t* restrict local = local_p;
  ptrdiff_t i;
  #pragma omp parallel for
  for (i = 0; i < (ptrdiff_t)count; ++i) {
    owner[i] = vertex[i]/nvertices_1d_per_proc(graph.lg_nvertices);
    local[i] = vertex[i]%nvertices_1d_per_proc(graph.lg_nvertices);
  }
}

int64_t vertex_to_global_for_pred(int v_rank, size_t v_local) {
  return v_local+v_rank*nvertices_1d_per_proc(graph.lg_nvertices);
}

size_t get_nlocalverts_for_pred(void) {
  return nvertices_1d_per_proc(graph.lg_nvertices);
}

int bfs_writes_depth_map(void) {
  return 0;
}
/* --------------------------------------------------------------------*/

using namespace std;

void init_gpu_pre()
{
 
}

void make_graph_data_structure(const tuple_graph* const tg)
{
	unsigned int omp_threads = 1;

	#pragma omp parallel 
	#pragma omp single 
	omp_threads=omp_get_num_threads();

	if(size != PROCESS_COUNT)
	{
		if(rank==0) fprintf(stderr,"Error PROCESS_COUNT != size \n");
		exit(EXIT_FAILURE);
	}

	graph.lg_nvertices = SCALE;
	bfs_rank = rank;
	line_rank = rank % (int)sqrt(PROCESS_COUNT) ;
	line_num = rank / (int)sqrt(PROCESS_COUNT);
	col_rank = rank / (int)sqrt(PROCESS_COUNT);
	col_num = rank % (int)sqrt(PROCESS_COUNT);

	// Représenter les blocks les plus petits = 32 sommets = 1 mot * 32 
	nblocks_line = BITMAP_UNIT(1) << ((int)lg_nvertices_2d_per_proc(graph.lg_nvertices)-(int)log2(BITMAP_UNIT_SIZE));

	// Communicateurs MPI
	MPI_Group world_group,line_group,col_group;

	// la ligne
	MPI_Comm_group(MPI_COMM_WORLD,&world_group);
	int * tmpRank = (int*)malloc(sizeof(int)*line_size);
	for(unsigned int i = 0 ; i < line_size ; ++i)
		tmpRank[i] = i+line_num*line_size;

	MPI_Group_incl(world_group,line_size,tmpRank,&line_group);
	MPI_Comm_create(MPI_COMM_WORLD,line_group,&mpi_line_comm);

	MPI_Group_rank(line_group,&line_comm_rank);

	// la colonne
	memset(tmpRank,0,sizeof(int)*col_size);
	for(unsigned int i = 0 ; i < col_size ; ++i)
		tmpRank[i] = i*col_size+col_num;

	MPI_Group_incl(world_group,col_size,tmpRank,&col_group);
	MPI_Comm_create(MPI_COMM_WORLD,col_group,&mpi_col_comm);

	free(tmpRank);
	MPI_Group_rank(col_group,&col_comm_rank);

	#ifdef DATA_GRAPH
	if(bfs_rank==0)
	{
		printf("Informations: \n");
		printf("   omp : %d\n",omp_threads);
		printf("   nglobaledges : %" PRId64 "\n",tg->nglobaledges);
		printf("   size : %d\n",size);
		printf("   bfs_rank : %d\n",bfs_rank);
		printf("   line_rank : %d\n",line_rank);
		printf("   line_num : %d\n",line_num);
		printf("   col_rank : %d\n",col_rank);
		printf("   col_num : %d\n",col_num);
		printf("   constant_size : %d\n",constant_size);
		printf("   lg_constant_size : %d\n",lg_constant_size);

		printf("   lg_nvertices : %d\n",graph.lg_nvertices);
		printf("   line_size : %d\n",line_size);	
		printf("   col_size : %d\n",col_size);
		printf("   nblocks : %" PRId64 "\n",nblocks(nblocks_line));
		printf("   nblock_line : %d\n",nblocks_line);
		printf("   nvertices : %d\n",nvertices(graph.lg_nvertices));
		printf("   lg_nvertices_1d_per_proc : %d\n",lg_nvertices_1d_per_proc(graph.lg_nvertices));
		printf("   nvertices_1d_per_proc : %d\n",nvertices_1d_per_proc(graph.lg_nvertices));
		printf("   lg_nvertices_2d_per_proc : %d\n",lg_nvertices_2d_per_proc(graph.lg_nvertices));
		printf("   nvertices_2d_per_proc : %d\n",nvertices_2d_per_proc(graph.lg_nvertices));
		printf("   nwords_per_proc : %d\n",nwords_per_proc_line(graph.lg_nvertices));
		printf("   nvertices_2d_per_block : %d\n",nvertices_2d_per_block(graph.lg_nvertices));
		printf("   nwords_per_block_line = %d\n",nwords_per_block_line(graph.lg_nvertices));
	}
	#endif

	// A partir de la liste des arêtes 
	// Compter les arêtes à envoyer
	int * nedge_send = (int*)xmalloc(sizeof(int)*constant_size);
	// Total d'arêtes pour chaque process
	int * nedge_total = (int*)xmalloc(sizeof(int)*constant_size);
	// Arêtes à recevoir de chaque process
	int * nedge_recv = (int*)xmalloc(sizeof(int)*constant_size);
	// Somme prefix pour le décalage des arêtes à recevoir
	int * nedge_recv_displ = (int*)xmalloc(sizeof(int)*(constant_size+1));
	// Somme prefix pour le décalage des arêtes à envoyer
	int * nedge_send_displ = (int*)xmalloc(sizeof(int)*(constant_size+1));
	// Tableau de case actuelle pour remplir les arêtes dans outgoing_edges
	int * nedge_send_offset = (int*)xmalloc(sizeof(int)*constant_size);

	uint64_t nedges = 0;
	uint64_t * incoming_edges = NULL;
		
	// 1. Compter les arêtes pour chaque voisin 
	#ifdef DATA_GRAPH
	double start_rep = MPI_Wtime();
	#endif
	ITERATE_TUPLE_GRAPH_BEGIN(tg,buf,bufsize)
	{

		memset(nedge_send,0,sizeof(int)*constant_size);
		memset(nedge_total,0,sizeof(int)*constant_size);
		memset(nedge_recv,0,sizeof(int)*constant_size);
		memset(nedge_recv_displ,0,sizeof(int)*(constant_size+1));
		memset(nedge_send_displ,0,sizeof(int)*(constant_size+1));
		memset(nedge_send_offset,0,sizeof(int)*constant_size);
		if(bfs_rank==0) printf("Getting vertex count block %zu / %zu \n",size_t(ITERATE_TUPLE_GRAPH_BLOCK_NUMBER),size_t(ITERATE_TUPLE_GRAPH_BLOCK_COUNT(tg)));

		#pragma omp parallel for
		for(unsigned int i = 0; i < bufsize ; ++i)
		{
			int owner;
			uint32_t v0 = (uint32_t)get_v0_from_edge(&buf[i]);
			uint32_t v1 = (uint32_t)get_v1_from_edge(&buf[i]);
			if(v0 == v1) continue;
			// On gère les arêtes dans les deux sens 
			owner = compute_edge_owner(v0,v1,graph.lg_nvertices);
			assert(owner < (int)constant_size);
			//printf("%u %u\n",v0,v1);
			#pragma omp atomic
			nedge_send[owner]++;

			owner = compute_edge_owner(v1,v0,graph.lg_nvertices);
			assert(owner < (int)constant_size);
			//printf("%u %u\n",v1,v0);
			#pragma omp atomic
			nedge_send[owner]++;
		}

		for(unsigned int i = 0 ; i < constant_size ; ++i)
			assert(nedge_send < 0 && "Error, overflow edges count\n");

		MPI_Alltoall(nedge_send,1,MPI_INT,nedge_recv,1,MPI_INT,MPI_COMM_WORLD);

		for(unsigned int i = 0; i < constant_size ; ++i)
		{
			nedge_recv_displ[i+1] = nedge_recv_displ[i] + nedge_recv[i];
			nedge_send_displ[i+1] = nedge_send_displ[i] + nedge_send[i];
		}

		#pragma omp parallel for 
		for(unsigned int i = 0 ; i < constant_size+1 ; ++i)
		{
			assert(nedge_recv_displ[i] < 0 && "Error, overflow edges count recv displ\n");
			assert(nedge_send_displ[i] < 0 && "Error, overflow edges count send displ\n");
		}

		// Tableau de reception et d'envoi => représentation des arête sur 64 bits 
		// utiliser un realloc(ptr,size) il agrandit mais préserve les données
		incoming_edges = (uint64_t*)xrealloc(incoming_edges,sizeof(uint64_t)*(nedge_recv_displ[constant_size]+nedges));
	
		uint64_t * outgoing_edges = (uint64_t*)xmalloc(sizeof(uint64_t)*nedge_send_displ[constant_size]);

		//3. Remplir le tableau d'envoi 
		if(bfs_rank==0) printf("Edges distribution block %zu / %zu \n",size_t(ITERATE_TUPLE_GRAPH_BLOCK_NUMBER),size_t(ITERATE_TUPLE_GRAPH_BLOCK_COUNT(tg)));
		
		#pragma omp parallel for 
		for(unsigned int i = 0 ; i < bufsize ; ++i)
		{
			int owner;
			size_t offset;
			uint32_t v0 = (uint32_t)get_v0_from_edge(&buf[i]);
			uint32_t v1 = (uint32_t)get_v1_from_edge(&buf[i]);
			if(v0 == v1) continue;
			// On gère l'arête dans les deux sens 
			owner = compute_edge_owner(v0,v1,graph.lg_nvertices);
			assert(owner < (int)constant_size && "Erreur owner");
			// Pour l'appel OpenMP, action atomique 
			offset = __sync_fetch_and_add(&nedge_send_offset[owner],1);
			//printf("%d : %u<>%u = %d\n",bfs_rank,v0,v1,owner);
			outgoing_edges[offset+nedge_send_displ[owner]] = encode_edge_uint64_t(v0,v1);

			owner = compute_edge_owner(v1,v0,graph.lg_nvertices);
			assert(owner < (int)constant_size && "Erreur owner");			
			offset = __sync_fetch_and_add(&nedge_send_offset[owner],1);
			//printf("%d : %u<>%u = %d\n",bfs_rank,v1,v0,owner);
			outgoing_edges[offset+nedge_send_displ[owner]] = encode_edge_uint64_t(v1,v0);
		}

		// 4. Envoi du tableau 
		MPI_Alltoallv(outgoing_edges,nedge_send,nedge_send_displ,MPI_UINT64_T,incoming_edges+nedges,nedge_recv,nedge_recv_displ,MPI_UINT64_T,MPI_COMM_WORLD);

		// Total des arêtes reçues 
		nedges += nedge_recv_displ[constant_size];
		xfree(outgoing_edges);

	} ITERATE_TUPLE_GRAPH_END;

	xfree(nedge_send);
	xfree(nedge_total);
	xfree(nedge_recv);
	xfree(nedge_recv_displ);
	xfree(nedge_send_displ);
	xfree(nedge_send_offset);

	#ifdef DATA_GRAPH
	double stop_rep = MPI_Wtime();
	#endif

	vector<uint32_t> * edge_list_CSR = new vector<uint32_t>[nvertices_2d_per_proc(graph.lg_nvertices)];
	vector<uint32_t> * edge_list_CSC = new vector<uint32_t>[nvertices_2d_per_proc(graph.lg_nvertices)];
	
	#ifdef DATA_GRAPH
	double start_csr_r = MPI_Wtime();
	#endif

	// Toutes aretes reçues mise en place du CSR
	//1 Compter pour le R 
	graph.CSR_R = (uint32_t *)xcalloc(nvertices_2d_per_proc(graph.lg_nvertices)+1,sizeof(uint32_t));
	graph.CSC_C = (uint32_t *)xcalloc(nvertices_2d_per_proc(graph.lg_nvertices)+1,sizeof(uint32_t));

	// Tableaux pour la répartition sur les threads openMP
	vector<uint32_t> ** threads_CSR = new vector<uint32_t>*[omp_threads];
	vector<uint32_t> ** threads_CSC = new vector<uint32_t>*[omp_threads];
	for(unsigned int i = 0 ; i < omp_threads ; ++i)
	{
		threads_CSR[i] = new vector<uint32_t>[nvertices_2d_per_proc(graph.lg_nvertices)];
		threads_CSC[i] = new vector<uint32_t>[nvertices_2d_per_proc(graph.lg_nvertices)];
	}


	#pragma omp parallel default(shared)
	{
		int idx = omp_get_thread_num();
		#pragma omp for
		for(unsigned int i = 0 ; i < nedges ; ++i)
		{
			// Et on prend les arêtes dans le bon sens pour le reste 
			uint32_t v0 = decode_v0_uint64_t(incoming_edges[i]);
			uint32_t v1 = decode_v1_uint64_t(incoming_edges[i]);
			uint32_t v0_proc = v0 % nvertices_2d_per_proc(graph.lg_nvertices);
			uint32_t v1_proc = v1 % nvertices_2d_per_proc(graph.lg_nvertices);
	
			if(line_num == col_num)
			{
				threads_CSR[idx][v0_proc].push_back(v1_proc);
				threads_CSC[idx][v1_proc].push_back(v0_proc);
			}

			threads_CSR[idx][v1_proc].push_back(v0_proc);
			threads_CSC[idx][v0_proc].push_back(v1_proc);
		}
	}

	xfree(incoming_edges);

	// On fait une série de merge en parallèle
	for(unsigned int i = 0 ; i < omp_threads ; ++i)
	{
		#pragma omp parallel for 
		for(unsigned int j = 0 ; j < nvertices_2d_per_proc(graph.lg_nvertices) ; ++j)
		{
			//printf("   tableau (%d,%d)\n",i,j);
			edge_list_CSR[j].insert(edge_list_CSR[j].end(),threads_CSR[i][j].begin(),threads_CSR[i][j].end());
			//printf("    .5\n");
			edge_list_CSC[j].insert(edge_list_CSC[j].end(),threads_CSC[i][j].begin(),threads_CSC[i][j].end());
		}
	}


// Pour chaque ligne compter les blocs et mettre en place CSR_R
	#ifdef DATA_GRAPH
	unsigned int useless_line = 0;
	#pragma omp parallel for reduction(+:useless_line)
	#else
	#pragma omp parallel for
	#endif
	for(unsigned int i = 0 ; i < nvertices_2d_per_proc(graph.lg_nvertices) ; ++i)	
	{
		//unique(edge_list_CSR[i].begin(),edge_list_CSR[i].end());
		__gnu_parallel::sort(edge_list_CSR[i].begin(),edge_list_CSR[i].end());
		__gnu_parallel::unique_copy(edge_list_CSR[i].begin(),edge_list_CSR[i].end(),edge_list_CSR[i].begin());
		graph.CSR_R[i]=edge_list_CSR[i].size();

		//unique(edge_list_CSC[i].begin(),edge_list_CSC[i].end());
		__gnu_parallel::sort(edge_list_CSC[i].begin(),edge_list_CSC[i].end());
		__gnu_parallel::unique_copy(edge_list_CSC[i].begin(),edge_list_CSC[i].end(),edge_list_CSC[i].begin());
		graph.CSC_C[i]=edge_list_CSC[i].size();

		#ifdef DATA_GRAPH
		if(graph.CSR_R[i] == 0)
			++useless_line;
		#endif
	}

	// Somme préfix pour le CSR 
	unsigned int tmp = graph.CSR_R[0];
	graph.CSR_R[0] = 0;
	for(unsigned int i = 1 ; i < nvertices_2d_per_proc(graph.lg_nvertices)+1 ; ++i)
	{
		unsigned int tmp2 = graph.CSR_R[i];
		graph.CSR_R[i] = graph.CSR_R[i-1] + tmp;
		tmp = tmp2;
	}

	// Somme préfix pour le CSC 
	tmp = graph.CSC_C[0];
	graph.CSC_C[0] = 0;
	for(unsigned int i = 1 ; i < nvertices_2d_per_proc(graph.lg_nvertices)+1 ; ++i)
	{
		unsigned int tmp2 = graph.CSC_C[i];
		graph.CSC_C[i] = graph.CSC_C[i-1] + tmp;
		tmp = tmp2;
	}

	#ifdef DATA_GRAPH
	double stop_csr_r = MPI_Wtime();
	double start_csr_c = MPI_Wtime();
	#endif

	//2 Remplir le C et les blocks
	graph.CSR_C = (uint32_t *)xcalloc(graph.CSR_R[nvertices_2d_per_proc(graph.lg_nvertices)],sizeof(uint32_t)); 
	
	#pragma omp parallel for
	for(unsigned int i = 0 ; i < nvertices_2d_per_proc(graph.lg_nvertices) ; ++i)	
	{
		unsigned int pos = 0;
		for(vector<uint32_t>::iterator it = edge_list_CSR[i].begin() ; it != edge_list_CSR[i].end(); ++it)
		{
			graph.CSR_C[graph.CSR_R[i]+pos] = *it;
			pos++;
		}
		edge_list_CSR[i].clear();
	}

	//2 Remplir le C et les blocks
	graph.CSC_R = (uint32_t *)xcalloc(graph.CSC_C[nvertices_2d_per_proc(graph.lg_nvertices)],sizeof(uint32_t)); 
	
	#pragma omp parallel for
	for(unsigned int i = 0 ; i < nvertices_2d_per_proc(graph.lg_nvertices) ; ++i)	
	{
		unsigned int pos = 0;
		for(vector<uint32_t>::iterator it = edge_list_CSC[i].begin() ; it != edge_list_CSC[i].end(); ++it)
		{
			graph.CSC_R[graph.CSC_C[i]+pos] = *it;
			pos++;
		}
		edge_list_CSC[i].clear();
	}

	delete[] edge_list_CSR;
	delete[] edge_list_CSC;

	#ifdef DATA_GRAPH
	double stop_csr_c = MPI_Wtime();
	#endif

	graph.in_queue = (BITMAP_UNIT*)xcalloc(nwords_per_proc_line(graph.lg_nvertices),sizeof(BITMAP_UNIT));
	graph.out_queue = (BITMAP_UNIT*)xcalloc(nwords_per_proc_line(graph.lg_nvertices),sizeof(BITMAP_UNIT));
	graph.my_assigned_targets = (BITMAP_UNIT*)xcalloc(nwords_per_proc_line(graph.lg_nvertices),sizeof(BITMAP_UNIT));
	MPI_Alloc_mem(sizeof(BITMAP_UNIT)*nwords_per_proc_line(graph.lg_nvertices),MPI_INFO_NULL,&graph.prefix); 
	graph.visited_v2 = (int32_t*)xmalloc(sizeof(uint32_t)*nvertices_2d_per_proc(graph.lg_nvertices));

	#ifdef DATA_GRAPH
	if(bfs_rank==0)
	{
		printf("Graph times : \n");
		printf(" Repartition :          %.4fs\n",stop_rep-start_rep);
		printf(" Graph_csr_r :          %.4fs\n",stop_csr_r-start_csr_r);
		printf(" Graph_csr_c :          %.4fs\n",stop_csr_c-start_csr_c);
		printf("Graph memory : \n");
		double taille = 0.f;
		printf(" csr_r :                %.4fGB\n",nvertices_2d_per_proc(graph.lg_nvertices)*sizeof(unsigned int)/1024.f/1024.f/1024.f);
		taille += nvertices_2d_per_proc(graph.lg_nvertices)*sizeof(unsigned int);
		printf(" csr_c :                %.4fGB\n",graph.CSR_R[nvertices_2d_per_proc(graph.lg_nvertices)]*sizeof(unsigned int)/1024.f/1024.f/1024.f);
		taille += graph.CSR_R[nvertices_2d_per_proc(graph.lg_nvertices)]*sizeof(unsigned int);
		//printf(" csr_v :                %.4fGB\n",graph.CSR_R[nblocks_line]*sizeof(BITMAP_UNIT)*BITMAP_UNIT_SIZE/1024.f/1024.f/1024.f);
		//taille += graph.CSR_R[nblocks_line]*sizeof(BITMAP_UNIT)*BITMAP_UNIT_SIZE;
		printf(" Total :                %.4fGB\n",taille/1024.f/1024.f/1024.f);
	
		printf("Stats : \n");
		printf(" useless_lines  :         %u\n",useless_line);
		printf("       percent  :         %.4f\n",useless_line*100.f/nvertices_2d_per_proc(graph.lg_nvertices)); 
		//printf(" nblocks_line :         %" PRId64 " \n",nblocks(nblocks_line));
		//printf(" nblobks_full :         %u \n",graph.CSR_R[nblocks_line]);
		//printf("      percent :         %.4f \n",(double)graph.CSR_R[nblocks_line]*100.f/(double)nblocks(nblocks_line));
		//printf(" nbits_set    :         %u \n",total_bit);
		//printf(" bit/block    :         %.4f \n",total_bit/(double)graph.CSR_R[nblocks_line]);
	}
	MPI_Barrier(MPI_COMM_WORLD);
	//exit(0);
	#endif
}

void free_graph_data_structure()
{
	//xfree(graph.block_size);
	xfree(graph.CSR_R);
	xfree(graph.CSR_C);
	xfree(graph.in_queue);
	xfree(graph.out_queue);
	xfree(graph.visited_v2);
	xfree(graph.my_assigned_targets);
	//xfree(graph.pos_in_queue);
	MPI_Free_mem(graph.prefix); 
}


void run_bfs(int64_t root, int64_t* pred)
{

	#pragma omp parallel
	#pragma omp single
	printf("OMP %d\n",omp_get_num_threads());

	#ifdef BFS_DATA
	double begin_time_start = MPI_Wtime();
	#endif

	#pragma omp parallel for
	for(unsigned int i = 0 ; i < nvertices_1d_per_proc(graph.lg_nvertices) ; ++i)
		pred[i] = -1;

	memset(graph.in_queue,0,nwords_per_proc_line(graph.lg_nvertices)*sizeof(BITMAP_UNIT));
	memset(graph.my_assigned_targets,0,nwords_per_proc_line(graph.lg_nvertices)*sizeof(BITMAP_UNIT));
	memset(graph.out_queue,0,nwords_per_proc_line(graph.lg_nvertices)*sizeof(BITMAP_UNIT));
	memset(graph.prefix,0,nwords_per_proc_line(graph.lg_nvertices)*sizeof(BITMAP_UNIT));

	#pragma omp parallel for 
	for(unsigned int i = 0 ; i < nvertices_2d_per_proc(graph.lg_nvertices) ; ++i)
		graph.visited_v2[i] = -1;

	unsigned int curlevel = 1;

	#ifdef BFS_DATA
	double steps_time[5] = {0.f,0.f,0.f,0.f,0.f};
	#endif

	/* Initialisation de ce bfs */
	// 1. La file d'entrée, visited et pred
	{
		unsigned int root_lineNcol = root / nvertices_2d_per_proc(graph.lg_nvertices);
		unsigned int root_word = (root % nvertices_2d_per_proc(graph.lg_nvertices)) / BITMAP_UNIT_SIZE;
		unsigned int root_bit = (root % nvertices_2d_per_proc(graph.lg_nvertices)) % BITMAP_UNIT_SIZE;
		if(root_lineNcol == col_num)
			graph.in_queue[root_word] = (uint32_t)(1) << root_bit;
		if(root_lineNcol == line_num)
			graph.visited_v2[root_word*BITMAP_UNIT_SIZE+root_bit] = root;

			//graph.visited[root_word] = (uint32_t)(1) << root_bit;

		int root_pred_owner = root / nvertices_1d_per_proc(graph.lg_nvertices);
		unsigned int root_pred_pos = (uint32_t)root % nvertices_1d_per_proc(graph.lg_nvertices);
		if(root_pred_owner == bfs_rank)
			pred[root_pred_pos] = root;
	}

	#ifdef BFS_DATA
	double begin_time_stop = MPI_Wtime();
	if(bfs_rank==0)printf("setup time : %.4f\n",begin_time_stop-begin_time_start);
	#endif


	while(1)
	{
		if(curlevel >= UINT16_MAX)
		{
			if(bfs_rank == 0) fprintf(stderr,"Trop d'itérations dans le bfs\n");
			exit(EXIT_FAILURE);
		}

		// étape 1, explore frontier
		#ifdef BFS_DATA
		double step_1_start = MPI_Wtime();
		#endif
		{
			memset(graph.out_queue,0,sizeof(BITMAP_UNIT)*nwords_per_proc_line(graph.lg_nvertices));
			graph.n_in_queue = 0;

			unsigned int sum = 0;

			if(graph.n_out_queue > nvertices_limit(graph.lg_nvertices))
			{
				printf("CSR");

				#pragma omp parallel for reduction(+:sum)
				for(unsigned int i = 0 ; i < (nvertices_2d_per_proc(graph.lg_nvertices)  / BITMAP_UNIT_SIZE) ; ++i)
				{
					for(unsigned int j = 0 ; j < BITMAP_UNIT_SIZE  ; ++j)
					{
						unsigned int ligne = i*BITMAP_UNIT_SIZE+j;
						if(graph.visited_v2[ligne] == -1)
						{
							int ok = 0;

							for(unsigned int k = graph.CSR_R[ligne] ; k < graph.CSR_R[ligne+1] && !ok; ++k)
							{
								if(graph.in_queue[graph.CSR_C[k]/BITMAP_UNIT_SIZE] & 1 << (graph.CSR_C[k]%BITMAP_UNIT_SIZE))
								{
									sum++;
									graph.out_queue[i] |= 1 << j; 
									graph.visited_v2[ligne] = graph.CSR_C[k] + col_num*nvertices_2d_per_proc(graph.lg_nvertices);
									ok = 1;
								}
							}
						}	
					}
				}		
			}else{

				printf("CSC");

				#pragma omp parallel for reduction(+:sum)
				for(unsigned int i = 0 ; i < nwords_per_proc_line(graph.lg_nvertices) ; ++i)
				{ 
					uint32_t val_in_queue = graph.in_queue[i];
					if(val_in_queue)
					{
						for(unsigned int j = 0 ; j < BITMAP_UNIT_SIZE ; ++j)
						{
							if(val_in_queue & 1 << j)
							{
								unsigned int colonne = i*BITMAP_UNIT_SIZE+j;
								for(unsigned int k = graph.CSC_C[colonne] ; k < graph.CSC_C[colonne+1] ; ++k)
								{
									if(graph.visited_v2[graph.CSC_R[k]] == -1)
									{
										++sum;
										#pragma omp atomic
										graph.out_queue[graph.CSC_R[k]/BITMAP_UNIT_SIZE] |= 1 << graph.CSC_R[k]%BITMAP_UNIT_SIZE;

										graph.visited_v2[graph.CSC_R[k]] = colonne + col_num*nvertices_2d_per_proc(graph.lg_nvertices);
									}
								}
							}
						}
					}
				}
			}

			graph.n_out_queue = sum;

		}
		#ifdef BFS_DATA
		double step_1_stop = MPI_Wtime();
		steps_time[0] += step_1_stop - step_1_start;
		printf(" %.4f \n",step_1_stop-step_1_start);
		//if(bfs_rank==0) printf("Step 1: %.4f\n",step_1_stop-step_1_start);
		#endif

		// Vérification de l'arrêt
		{
			MPI_Allreduce(MPI_IN_PLACE,&(graph.n_out_queue),1,MPI_UINT32_T,MPI_SUM,MPI_COMM_WORLD);
			#ifdef BFS_DATA
			if(bfs_rank==0) printf("level %d => %u vertices\n",curlevel,graph.n_out_queue);
			#endif
			if(graph.n_out_queue==0)
			{
				if(bfs_rank == 0) printf("bfs end level %d\n",curlevel);
				break;
			}
		}	

		// étape 2, échange en ligne 
		#ifdef BFS_DATA
		double step_2_start = MPI_Wtime();
		#endif
		{
			//memset(graph.my_assigned_targets,0,sizeof(uint32_t)*nwords_per_proc_line(graph.lg_nvertices));
			#if PROCESS_COUNT == 1
			memset(graph.prefix,0,sizeof(BITMAP_UNIT)*nwords_per_proc_line(graph.lg_nvertices));
			MPI_Exscan(graph.out_queue,graph.prefix,nwords_per_proc_line(graph.lg_nvertices),BITMAP_UNIT_MPI,MPI_BOR,mpi_line_comm);
			#else
			memcpy(graph.in_queue,graph.out_queue,nwords_per_proc_line(graph.lg_nvertices)*sizeof(BITMAP_UNIT));
			//On utilise la file d'entrée comme buffer de reception
			MPI_Status stat;
			if(line_rank != 0)
			{
				MPI_Recv(graph.prefix,nwords_per_proc_line(graph.lg_nvertices),BITMAP_UNIT_MPI,line_rank-1,0,mpi_line_comm,&stat);
				#pragma omp parallel for 
				for(unsigned int i = 0 ; i < nwords_per_proc_line(graph.lg_nvertices) ; ++i)
					graph.in_queue[i] |= graph.prefix[i];
			}else{
				MPI_Send(graph.in_queue,nwords_per_proc_line(graph.lg_nvertices),BITMAP_UNIT_MPI,line_rank+1,0,mpi_line_comm);
				memset(graph.prefix,0,sizeof(BITMAP_UNIT)*nwords_per_proc_line(graph.lg_nvertices));
			}

			if(line_rank != 0 && line_rank != line_size-1)
				MPI_Send(graph.in_queue,nwords_per_proc_line(graph.lg_nvertices),BITMAP_UNIT_MPI,line_rank+1,0,mpi_line_comm);

			#endif
			#pragma omp parallel for
			for(unsigned int i = 0 ; i < nwords_per_proc_line(graph.lg_nvertices) ; ++i)
				graph.my_assigned_targets[i] |= (graph.out_queue[i] & graph.prefix[i]) ^ graph.out_queue[i];		
		}
		#ifdef BFS_DATA
		double step_2_stop = MPI_Wtime();
		steps_time[1] += step_2_stop - step_2_start;
		//if(bfs_rank==0) printf("Step 2: %.4f\n",step_2_stop-step_2_start);
		#endif

		// étape 3, échange outQ
		// Le proc en bout de ligne génère la outQ générale 
		#ifdef BFS_DATA
		double step_3_start = MPI_Wtime();
		#endif
		{
			if(col_num == (line_size-1))
			{
				#pragma omp parallel for
				for(unsigned int i = 0 ; i < nwords_per_proc_line(graph.lg_nvertices) ; ++i)
					graph.out_queue[i] |= graph.prefix[i];
			}
			MPI_Bcast(graph.out_queue,nwords_per_proc_line(graph.lg_nvertices),BITMAP_UNIT_MPI,line_size-1,mpi_line_comm);

			/*#pragma omp parallel for 
			for(unsigned int i = 0 ; i < nwords_per_proc_line(graph.lg_nvertices) ; ++i)
			{
				graph.visited[i] |= graph.out_queue[i];
			}*/

			/* Mise à jour des visites */
			#pragma omp parallel for 
			for(unsigned int i = 0 ; i <  nwords_per_proc_line(graph.lg_nvertices); ++i)
			{
				for(unsigned int j = 0 ; j < nvertices_2d_per_block(graph.lg_nvertices) ; ++j)
					if(graph.out_queue[i] & 1 << j && graph.visited_v2[i*nvertices_2d_per_block(graph.lg_nvertices)+j]==-1)
					{
						graph.visited_v2[i*nvertices_2d_per_block(graph.lg_nvertices)+j] = -2;
					}
			}
		}
		#ifdef BFS_DATA
		double step_3_stop = MPI_Wtime();
		steps_time[2] += step_3_stop - step_3_start;
		//if(bfs_rank==0) printf("Step 3: %.4f\n",step_3_stop-step_3_start);
		#endif

		// étape 4, write pred	
		#ifdef BFS_DATA
		double step_4_start = MPI_Wtime();
		#endif
		{
			
		}
		#ifdef BFS_DATA
		double step_4_stop = MPI_Wtime();
		steps_time[3] += step_4_stop - step_4_start;
		//if(bfs_rank==0) printf("Step 4: %.4f\n",step_4_stop-step_4_start);
		#endif

		// étape 5, échange inQ et outQ
		#ifdef BFS_DATA
		double step_5_start = MPI_Wtime();
		#endif
		{
			memcpy(graph.in_queue,graph.out_queue,sizeof(BITMAP_UNIT)*nwords_per_proc_line(graph.lg_nvertices));
			MPI_Bcast(graph.in_queue,nwords_per_proc_line(graph.lg_nvertices),BITMAP_UNIT_MPI,col_num,mpi_col_comm);
		}

		#ifdef BFS_DATA
		double step_5_stop = MPI_Wtime();
		steps_time[4] += step_5_stop - step_5_start;
		//if(bfs_rank==0) printf("Step 5: %.4f\n",step_5_stop-step_5_start);
		#endif
		++curlevel;
	}


	#ifdef BFS_DATA
	double step_pred_start = MPI_Wtime();
	#endif

	/* Mise à jour des prédecesseurs */
	#pragma omp parallel for 
	for(unsigned int i = 0 ; i < nwords_per_proc_line(graph.lg_nvertices) ; ++i)
	{
		for(unsigned int j = 0 ; j < nvertices_2d_per_block(graph.lg_nvertices) ; ++j)
		{
			if((graph.visited_v2[i*nvertices_2d_per_block(graph.lg_nvertices)+j] != -1) && !(graph.my_assigned_targets[i] & 1 << j))
			{
				graph.visited_v2[i*nvertices_2d_per_block(graph.lg_nvertices)+j] = -1;
			}
			if(graph.visited_v2[i*nvertices_2d_per_block(graph.lg_nvertices)+j] == -2)
			{
				graph.visited_v2[i*nvertices_2d_per_block(graph.lg_nvertices)+j] = -1;
			}
		}
	}

	// Echanger tableaux sur la ligne 
	MPI_Alltoall(MPI_IN_PLACE,nvertices_1d_per_proc(graph.lg_nvertices),MPI_INT,graph.visited_v2,nvertices_1d_per_proc(graph.lg_nvertices),MPI_INT,mpi_line_comm);

	// Mettre à jour les prédecesseurs de sa liste
	#pragma omp parallel for 
	for(unsigned int j = 0 ; j < nvertices_2d_per_proc(graph.lg_nvertices) ; ++j)
	{
		if(graph.visited_v2[j] != -1 && pred[j%nvertices_1d_per_proc(graph.lg_nvertices)] == -1)
		{			
			pred[j%nvertices_1d_per_proc(graph.lg_nvertices)] = (int64_t)graph.visited_v2[j];
		}
	}



	#ifdef BFS_DATA
	double step_pred_stop = MPI_Wtime();
	if(bfs_rank==0) printf("Step pred: %.4f\n",step_pred_stop-step_pred_start);
	#endif


	#ifdef BFS_DATA
	if(bfs_rank==0)
	{
		double sum = 0;
		for(unsigned int i = 0 ; i < 5 ; ++i)
		{
				//steps_time[i] /= curlevel;
				sum += steps_time[i];
				printf("step %d : %.4f s\n",i,steps_time[i]);
		}
		steps_time[3] = step_pred_stop-step_pred_start;
		sum += steps_time[3];
		for(unsigned int i = 0 ; i < 5 ; ++i)
		{
				printf("step %d : %.4f s => %.4f percents \n",i,steps_time[i],steps_time[i]*100/sum);
		}
	}
	fflush(stdout);

	#endif
}
