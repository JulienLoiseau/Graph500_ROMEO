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


#include <iostream>
#include <list>

#define NWARPS 4
#define MIN_BLOCKS_PER_SMX 16

static void HandleError( cudaError_t err, const char * file, int line)
{
	if(err != cudaSuccess)
	{
		printf("%s dans %s en ligne line %d\n",cudaGetErrorString(err),file,line);
		exit(EXIT_FAILURE);
	}
}

#define HANDLE_ERROR(err) (HandleError(err,__FILE__,__LINE__))


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
static const unsigned int lg_constant_size = log2((float)constant_size);
static const unsigned int line_size = sqrt(constant_size);
static const unsigned int col_size = line_size;
static const unsigned int lg_line_size = lg_constant_size / 2;


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

static inline unsigned int nwords_per_proc_line(const unsigned int lg_nvertices)
{
	// Ici on round up pour les cas avec peu de sommets 
	return (nvertices_2d_per_proc(lg_nvertices) + BITMAP_UNIT_SIZE  -1 )/ BITMAP_UNIT_SIZE;
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
	int32_t * visited_label;

 	uint32_t n_out_queue;
 	
 	BITMAP_UNIT * in_queue;
 	BITMAP_UNIT * out_queue;

	unsigned int * CSC_C;
	unsigned int * CSC_R;

	unsigned int * CSR_C;
	unsigned int * CSR_R;

	unsigned int lg_nvertices; /* Global count */	
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

__constant__ unsigned int d_nvertices_1d_per_proc;
__constant__ unsigned int d_nvertices_2d_per_proc;
__constant__ unsigned int d_bfs_rank;
__constant__ unsigned int d_col_num;
__constant__ unsigned int d_offset;
__constant__ unsigned int d_line_num;

unsigned int * d_CSC_R;
unsigned int * d_CSC_C;

unsigned int * d_CSR_R;
unsigned int * d_CSR_C;

BITMAP_UNIT * d_in_queue;
BITMAP_UNIT * d_out_queue;
BITMAP_UNIT * d_assigned;
BITMAP_UNIT * d_prefix;
BITMAP_UNIT * d_visited_tex;
int32_t * d_visited_label;
uint32_t * d_n_out_queue;
int32_t * d_visited_label_tmp;

int64_t * d_pred;

texture<BITMAP_UNIT, 1, cudaReadModeElementType> tex_visited;
texture<BITMAP_UNIT, 1, cudaReadModeElementType> tex_in_queue;

void init_gpu_pre()
{
	int local_rank = atoi(getenv("MV2_COMM_WORLD_LOCAL_RANK")); 
	//HANDLE_ERROR(cudaSetDevice(local_rank % 2)); 

	HANDLE_ERROR(cudaSetDevice(0)); 
}


// Allocation mémoire C,R et blocks
void init_gpu(uint32_t* CSC_C,uint32_t* CSC_R, uint32_t * CSR_R, uint32_t * CSR_C)
{	

	// Informations nécessaires au GPU 
	HANDLE_ERROR(cudaMalloc(&d_in_queue,sizeof(BITMAP_UNIT)*nwords_per_proc_line(graph.lg_nvertices)));
	HANDLE_ERROR(cudaMalloc(&d_out_queue,sizeof(BITMAP_UNIT)*nwords_per_proc_line(graph.lg_nvertices)));
	HANDLE_ERROR(cudaMalloc(&d_prefix,sizeof(BITMAP_UNIT)*nwords_per_proc_line(graph.lg_nvertices)));
	HANDLE_ERROR(cudaMalloc(&d_assigned,sizeof(BITMAP_UNIT)*nwords_per_proc_line(graph.lg_nvertices)));
	// Visited en binaire 
	HANDLE_ERROR(cudaMalloc(&d_visited_tex,sizeof(BITMAP_UNIT)*nwords_per_proc_line(graph.lg_nvertices)));
	

	HANDLE_ERROR(cudaMalloc(&d_visited_label,sizeof(int32_t)*nvertices_2d_per_proc(graph.lg_nvertices)));
	HANDLE_ERROR(cudaMalloc(&d_visited_label_tmp,sizeof(int32_t)*nvertices_2d_per_proc(graph.lg_nvertices)));
	HANDLE_ERROR(cudaMalloc(&d_n_out_queue,sizeof(uint32_t)));

	HANDLE_ERROR(cudaMalloc(&d_CSC_C,sizeof(uint32_t)*(nvertices_2d_per_proc(graph.lg_nvertices)+1)));
	HANDLE_ERROR(cudaMalloc(&d_CSC_R,sizeof(uint32_t)* CSC_C[nvertices_2d_per_proc(graph.lg_nvertices)]));
	
	HANDLE_ERROR(cudaMalloc(&d_CSR_R,sizeof(uint32_t)*(nvertices_2d_per_proc(graph.lg_nvertices)+1)));
	HANDLE_ERROR(cudaMalloc(&d_CSR_C,sizeof(uint32_t)* CSR_R[nvertices_2d_per_proc(graph.lg_nvertices)]));
	
	/* Copie en mémoire de la matrice */
	HANDLE_ERROR(cudaMemcpy(d_CSC_C,CSC_C,sizeof(uint32_t)*(nvertices_2d_per_proc(graph.lg_nvertices)+1),cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(d_CSC_R,CSC_R,sizeof(uint32_t)*CSC_C[nvertices_2d_per_proc(graph.lg_nvertices)],cudaMemcpyHostToDevice));
	
	/* Copie en mémoire de la matrice */
	HANDLE_ERROR(cudaMemcpy(d_CSR_R,CSR_R,sizeof(uint32_t)*(nvertices_2d_per_proc(graph.lg_nvertices)+1),cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(d_CSR_C,CSR_C,sizeof(uint32_t)*CSR_R[nvertices_2d_per_proc(graph.lg_nvertices)],cudaMemcpyHostToDevice));

	HANDLE_ERROR(cudaMalloc(&d_pred,sizeof(int64_t)*nvertices_1d_per_proc(graph.lg_nvertices)));

	/* Copies données constantes */
	unsigned int tmp = nvertices_2d_per_proc(graph.lg_nvertices);
	HANDLE_ERROR(cudaMemcpyToSymbol(d_nvertices_2d_per_proc,&tmp,sizeof(unsigned int)));
	tmp = nvertices_1d_per_proc(graph.lg_nvertices);
	HANDLE_ERROR(cudaMemcpyToSymbol(d_nvertices_1d_per_proc,&tmp,sizeof(unsigned int)));
	tmp = nvertices_2d_per_proc(graph.lg_nvertices)*col_num;
	HANDLE_ERROR(cudaMemcpyToSymbol(d_offset,&tmp,sizeof(unsigned int)));
	HANDLE_ERROR(cudaMemcpyToSymbol(d_col_num,&col_num,sizeof(unsigned int)));
	HANDLE_ERROR(cudaMemcpyToSymbol(d_line_num,&line_num,sizeof(unsigned int)));
	HANDLE_ERROR(cudaMemcpyToSymbol(d_bfs_rank,&bfs_rank,sizeof(unsigned int)));
}


void make_graph_data_structure(const tuple_graph* const tg)
{

	if(size != PROCESS_COUNT)
	{
		if(rank==0) fprintf(stderr,"Error PROCESS_COUNT != size \n");
		exit(EXIT_FAILURE);
	}

	HANDLE_ERROR(cudaSetDevice(0));
	//HANDLE_ERROR(cudaSetDevice(rank % 2));
	
	graph.lg_nvertices = SCALE;
	bfs_rank = rank;
	line_rank = rank % (int)sqrt(PROCESS_COUNT) ;
	line_num = rank / (int)sqrt(PROCESS_COUNT);
	col_rank = rank / (int)sqrt(PROCESS_COUNT);
	col_num = rank % (int)sqrt(PROCESS_COUNT);

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
		#pragma omp parallel 
		#pragma omp single
		printf("   omp : %d\n",omp_get_num_threads());
		
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
		printf("   nvertices : %d\n",nvertices(graph.lg_nvertices));
		printf("   lg_nvertices_1d_per_proc : %d\n",lg_nvertices_1d_per_proc(graph.lg_nvertices));
		printf("   nvertices_1d_per_proc : %d\n",nvertices_1d_per_proc(graph.lg_nvertices));
		printf("   lg_nvertices_2d_per_proc : %d\n",lg_nvertices_2d_per_proc(graph.lg_nvertices));
		printf("   nvertices_2d_per_proc : %d\n",nvertices_2d_per_proc(graph.lg_nvertices));
		printf("   nwords_per_proc : %d\n",nwords_per_proc_line(graph.lg_nvertices));
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

	uint64_t * incoming_edges = NULL;

	// Générer la répartition ligne/colonne au fur et à mesure 
	// On prend toute la largeur de la colonne (CSC)
	list<uint32_t> * edge_list_GPU = new list<uint32_t>[nvertices_2d_per_proc(graph.lg_nvertices)];
	// On prend seulement les lignes interessantes pour le GPU
	list<uint32_t> * edge_list_CPU = new list<uint32_t>[nvertices_2d_per_proc(graph.lg_nvertices)];
	
	if(!edge_list_GPU || !edge_list_CPU)
	{
		fprintf(stderr,"Erreur allocation edge_list\n");
		exit(0);
	}
	// Taille de CSR_R connue = nSommets +1
	graph.CSC_C = (uint32_t *)xcalloc(nvertices_2d_per_proc(graph.lg_nvertices)+1,sizeof(uint32_t));
	
	//Attention à la taille 
	graph.CSR_R = (uint32_t *)xcalloc(nvertices_2d_per_proc(graph.lg_nvertices)+1,sizeof(uint32_t));

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
			//printf("%u %u\n",v0,v1);
			#pragma omp atomic
			nedge_send[owner]++;

			owner = compute_edge_owner(v1,v0,graph.lg_nvertices);
			//printf("%u %u\n",v1,v0);
			#pragma omp atomic
			nedge_send[owner]++;
		}

		MPI_Alltoall(nedge_send,1,MPI_INT,nedge_recv,1,MPI_INT,MPI_COMM_WORLD);

		for(unsigned int i = 0; i < constant_size ; ++i)
		{
			nedge_recv_displ[i+1] = nedge_recv_displ[i] + nedge_recv[i];
			nedge_send_displ[i+1] = nedge_send_displ[i] + nedge_send[i];
		}

		// Tableau de reception et d'envoi => représentation des arête sur 64 bits 
		// utiliser un realloc(ptr,size) il agrandit mais préserve les données
		incoming_edges = (uint64_t*)xmalloc(sizeof(uint64_t)*(nedge_recv_displ[constant_size]));
	
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
			// Pour l'appel OpenMP, action atomique 
			offset = __sync_fetch_and_add(&nedge_send_offset[owner],1);
			//printf("%d : %u<>%u = %d\n",bfs_rank,v0,v1,owner);
			outgoing_edges[offset+nedge_send_displ[owner]] = encode_edge_uint64_t(v0,v1);

			owner = compute_edge_owner(v1,v0,graph.lg_nvertices);
			offset = __sync_fetch_and_add(&nedge_send_offset[owner],1);
			//printf("%d : %u<>%u = %d\n",bfs_rank,v1,v0,owner);
			outgoing_edges[offset+nedge_send_displ[owner]] = encode_edge_uint64_t(v1,v0);
		}

		// 4. Envoi du tableau 
		MPI_Alltoallv(outgoing_edges,nedge_send,nedge_send_displ,MPI_UINT64_T,incoming_edges,nedge_recv,nedge_recv_displ,MPI_UINT64_T,MPI_COMM_WORLD);
		
		// On ajoute les aretes de ce paquet
		for(unsigned int i = 0 ; i < (unsigned int)nedge_recv_displ[constant_size] ; ++i)
		{
			// Et on prend les arêtes dans le bon sens pour le reste 
			uint32_t v0 = decode_v0_uint64_t(incoming_edges[i]);
			uint32_t v1 = decode_v1_uint64_t(incoming_edges[i]);
			uint32_t v0_proc = v0 % nvertices_2d_per_proc(graph.lg_nvertices);
			uint32_t v1_proc = v1 % nvertices_2d_per_proc(graph.lg_nvertices);
			
			edge_list_GPU[v0_proc].insert(edge_list_GPU[v0_proc].begin(),v1_proc);
			edge_list_CPU[v1_proc].insert(edge_list_CPU[v1_proc].begin(),v0_proc);
			
			// Pareil pour l'arete dans l'autre sens v1 => v0
			if(line_num == col_num)
			{
				edge_list_GPU[v1_proc].insert(edge_list_GPU[v1_proc].begin(),v0_proc);
				edge_list_CPU[v0_proc].insert(edge_list_CPU[v0_proc].begin(),v1_proc);
			}
		}
		#pragma omp parallel for
		for(unsigned int i = 0 ; i < nvertices_2d_per_proc(graph.lg_nvertices) ; ++i)	
		{
			edge_list_GPU[i].unique();
			edge_list_GPU[i].sort();
			edge_list_CPU[i].unique();
			edge_list_CPU[i].sort();
		}
	
		xfree(outgoing_edges);
		xfree(incoming_edges);

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

	#ifdef DATA_GRAPH
		double start_csr_r = MPI_Wtime();
	#endif

	// Pour chaque ligne compter les blocs et mettre en place CSR_R
	#pragma omp parallel for
	for(unsigned int i = 0 ; i < nvertices_2d_per_proc(graph.lg_nvertices) ; ++i)	
	{
		graph.CSC_C[i]=edge_list_GPU[i].size();
	}
	
	#pragma omp parallel for
	for(unsigned int i = 0 ; i < nvertices_2d_per_proc(graph.lg_nvertices) ; ++i)	
	{
		graph.CSR_R[i]=edge_list_CPU[i].size();
	}

	// Utilisation du GPU pour faire la répartition des arêtes 

	// Somme préfix pour le CSC 
	unsigned int tmp = graph.CSC_C[0];
	graph.CSC_C[0] = 0;
	for(unsigned int i = 1 ; i < nvertices_2d_per_proc(graph.lg_nvertices)+1 ; ++i)
	{
		unsigned int tmp2 = graph.CSC_C[i];
		graph.CSC_C[i] = graph.CSC_C[i-1] + tmp;
		tmp = tmp2;
	}

	// Somme préfix pour le CSR 
	tmp = graph.CSR_R[0];
	graph.CSR_R[0] = 0;
	for(unsigned int i = 1 ; i < nvertices_2d_per_proc(graph.lg_nvertices)+1 ; ++i)
	{
		unsigned int tmp2 = graph.CSR_R[i];
		graph.CSR_R[i] = graph.CSR_R[i-1] + tmp;
		tmp = tmp2;
	}

#ifdef DATA_GRAPH
	printf("Aretes GPU %d CPU %d\n",graph.CSC_C[nvertices_2d_per_proc(graph.lg_nvertices)],graph.CSR_R[nvertices_2d_per_proc(graph.lg_nvertices)]);
#endif

	#ifdef DATA_GRAPH
	double stop_csr_r = MPI_Wtime();
	double start_csr_c = MPI_Wtime();
	#endif

#ifdef DATA_GRAPH
	printf("Nb moyen d'elements par ligne : %d\n",graph.CSC_C[nvertices_2d_per_proc(graph.lg_nvertices)]/nvertices_2d_per_proc(graph.lg_nvertices));
#endif

	//2 Remplir le C et les blocks
	graph.CSC_R = (uint32_t *)xcalloc(graph.CSC_C[nvertices_2d_per_proc(graph.lg_nvertices)],sizeof(uint32_t)); 
	graph.CSR_C = (uint32_t *)xcalloc(graph.CSR_R[nvertices_2d_per_proc(graph.lg_nvertices)],sizeof(uint32_t)); 

	#pragma omp parallel for
	for(unsigned int i = 0 ; i < nvertices_2d_per_proc(graph.lg_nvertices) ; ++i)	
	{
		unsigned int pos = 0;
		for(list<uint32_t>::iterator it = edge_list_GPU[i].begin() ; it != edge_list_GPU[i].end(); ++it)
		{
			graph.CSC_R[graph.CSC_C[i]+pos] = *it;
			pos++;
		}
		edge_list_GPU[i].clear();
	}

	#pragma omp parallel for
	for(unsigned int i = 0 ; i < nvertices_2d_per_proc(graph.lg_nvertices) ; ++i)	
	{
		unsigned int pos = 0;
		for(list<uint32_t>::iterator it = edge_list_CPU[i].begin() ; it != edge_list_CPU[i].end(); ++it)
		{
			graph.CSR_C[graph.CSR_R[i]+pos] = *it;
			pos++;
		}
		edge_list_CPU[i].clear();
	}

	delete[] edge_list_GPU;
	delete[] edge_list_CPU;

	#ifdef DATA_GRAPH
	double stop_csr_c = MPI_Wtime();
	#endif


	graph.visited_label = (int32_t*)xmalloc(sizeof(int32_t)*nvertices_2d_per_proc(graph.lg_nvertices));
	graph.in_queue = (BITMAP_UNIT*)xcalloc(nwords_per_proc_line(graph.lg_nvertices),sizeof(BITMAP_UNIT));
	graph.out_queue = (BITMAP_UNIT*)xcalloc(nwords_per_proc_line(graph.lg_nvertices),sizeof(BITMAP_UNIT));
	/*graph.my_assigned_targets = (BITMAP_UNIT*)xcalloc(nwords_per_proc_line(graph.lg_nvertices),sizeof(BITMAP_UNIT));
	
	MPI_Alloc_mem(sizeof(BITMAP_UNIT)*nwords_per_proc_line(graph.lg_nvertices),MPI_INFO_NULL,&graph.prefix);*/

	/*if(!graph.prefix)
	{
		fprintf(stderr,"Erreur malloc graphe.prefix\n");
		exit(0);
	}*/

	// Chargement en mémoire GPU
	init_gpu(graph.CSC_C,graph.CSC_R,graph.CSR_R,graph.CSR_C);

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
		printf(" csr_c :                %.4fGB\n",graph.CSC_R[nvertices_2d_per_proc(graph.lg_nvertices)]*sizeof(unsigned int)/1024.f/1024.f/1024.f);
		taille += graph.CSC_R[nvertices_2d_per_proc(graph.lg_nvertices)]*sizeof(unsigned int);
		printf(" Total :                %.4fGB\n",taille/1024.f/1024.f/1024.f);
	}
	MPI_Barrier(MPI_COMM_WORLD);
	#endif
}

void free_graph_data_structure()
{
	cudaDeviceSynchronize();
	HANDLE_ERROR(cudaFree(d_in_queue));
	HANDLE_ERROR(cudaFree(d_out_queue));
	HANDLE_ERROR(cudaFree(d_n_out_queue));
	HANDLE_ERROR(cudaFree(d_visited_label));
	HANDLE_ERROR(cudaFree(d_visited_label_tmp));
	HANDLE_ERROR(cudaFree(d_visited_tex));
	HANDLE_ERROR(cudaFree(d_assigned));
	HANDLE_ERROR(cudaFree(d_CSC_R));
	HANDLE_ERROR(cudaFree(d_CSC_C));
	xfree(graph.CSR_R);
	xfree(graph.CSR_C);
	xfree(graph.CSC_R);
	xfree(graph.CSC_C);
	xfree(graph.visited_label);
}

__global__ void setup_bfs_kernel(int32_t root, BITMAP_UNIT * in_queue, int32_t * visited_label, BITMAP_UNIT * visited_tex, int64_t * pred)
{
	unsigned int root_lineNcol = root / d_nvertices_2d_per_proc;
	unsigned int root_word = (root % d_nvertices_2d_per_proc) / BITMAP_UNIT_SIZE;
	unsigned int root_bit = (root % d_nvertices_2d_per_proc) % BITMAP_UNIT_SIZE;
	if(root_lineNcol == d_col_num)
		in_queue[root_word] = (BITMAP_UNIT)(1) << root_bit;
	if(root_lineNcol == d_line_num)
	{
		visited_label[root_word*BITMAP_UNIT_SIZE+root_bit] = root;
		//initialiser la visited en texture 
		visited_tex[root_word] = (BITMAP_UNIT)(1) << root_bit;
	}
	int root_pred_owner = root / d_nvertices_1d_per_proc;
	unsigned int root_pred_pos = (uint32_t)root % d_nvertices_1d_per_proc;
	if(root_pred_owner == d_bfs_rank)
	{
		pred[root_pred_pos] = root;
	}
}

__global__ void raz_visited_pred_kernel(int32_t * visited_label, int64_t * pred)
{
	unsigned int id = threadIdx.x+blockIdx.x*blockDim.x;
	visited_label[id] = -1;
	if(id < d_nvertices_1d_per_proc)
		pred[id] = -1;
}

__inline__ __device__
int warpReduceSum(int val) {
  for (int offset = warpSize/2; offset > 0; offset /= 2) 
    val += __shfl_down(val, offset);
  return val;
}

__launch_bounds__(NWARPS*32, MIN_BLOCKS_PER_SMX)
__global__ void explore_frontier_CSR_warps_v2( BITMAP_UNIT * out_queue,  int32_t * visited_label, BITMAP_UNIT * visited_tex,  uint32_t * n_out_queue, unsigned int * R, unsigned int * C)
{
	int lane_id = threadIdx.x & 31;
	int warp_id = threadIdx.x >> 5;
	
	int ligne = threadIdx.x+blockIdx.x*blockDim.x;
	int32_t value_visited = visited_label[ligne];			//GLOBAL
	int actif = 0;
	if(value_visited == -1)
		actif = 1;
		
	if(!__any(actif))
		return;
		
	unsigned int word = ligne/BITMAP_UNIT_SIZE;
	unsigned int range[3] = {0,0,0};
	
	if(value_visited == -1)
	{
		range[0] = R[ligne];
		range[1] = R[ligne+1];
		range[2] = range[1] - range[0];
	}
	
	// On va explorer chaque ligne successivement 
	volatile __shared__ int comm[NWARPS][3];
	volatile __shared__ int shared_ligne[NWARPS];
	volatile __shared__ int sum[NWARPS];
	volatile __shared__ int fin[NWARPS];
	
	if(lane_id == 0)
		sum[warp_id] = 0;
	
	while( __any(range[2]) )
	{
		int voisin = -1;
	
		if(range[2])
			comm[warp_id][0] = lane_id;
	
		if(comm[warp_id][0] == lane_id)
		{
			comm[warp_id][1] = range[0];
			comm[warp_id][2] = range[1];
			range[2] = 0;
			shared_ligne[warp_id] = ligne;
		}
		
		int r_gather = comm[warp_id][1] + lane_id;
		int r_gather_end = comm[warp_id][2];
		
		if(lane_id==0)
			fin[warp_id] = 0;
		
		while(r_gather < r_gather_end && !fin[warp_id])
		{
			voisin = C[r_gather];
	
			// Vérifier voisin dans in_queue
			unsigned int position = voisin / BITMAP_UNIT_SIZE;
			BITMAP_UNIT mask = tex1Dfetch(tex_in_queue,position);
			BITMAP_UNIT mask_bit = 1 << (voisin % BITMAP_UNIT_SIZE);
			if(mask & mask_bit)
			{
				// Ajout direct du voisin dans visited et passer à la suite 
				//visited_label[shared_ligne[warp_id]] = voisin+d_offset;
				//int old = atomicCAS(&visited_label[shared_ligne[warp_id]],-1,voisin+d_offset);
				//if(old == -1)

				visited_label[shared_ligne[warp_id]] =  voisin+d_offset;
				if(visited_label[shared_ligne[warp_id]] == voisin+d_offset)
				{
					visited_tex[word] |= 1 << shared_ligne[warp_id]%BITMAP_UNIT_SIZE;
					out_queue[word] |= 1 << shared_ligne[warp_id]%BITMAP_UNIT_SIZE;
					++sum[warp_id];
					fin[warp_id] = 1;
				}
			}
			r_gather+=32;
		}
	}
	
	if(lane_id == 0 && sum[warp_id])
		atomicAdd(n_out_queue,sum[warp_id]);
}

__launch_bounds__(NWARPS*32, MIN_BLOCKS_PER_SMX)
__global__ void explore_frontier_CSC_warps_v2( restrict BITMAP_UNIT * in_queue, restrict BITMAP_UNIT * out_queue,  int32_t * visited_label, BITMAP_UNIT * visited_tex , uint32_t * n_out_queue, unsigned int * R, unsigned int * C)
{
	int lane_id = threadIdx.x & 31;
	int warp_id = threadIdx.x >> 5; 

	int word = blockIdx.x*NWARPS+warp_id;
	int val_in_queue = in_queue[word];								// GLOBAL
	if(val_in_queue == 0)
		return;

	int id_sommet = -1;
	unsigned int range[3] = {0,0,0};
	
	if(val_in_queue & 1 << lane_id)
	{
		id_sommet = word*32+lane_id;
		range[0] = C[id_sommet];									//GLOBAL
		range[1] = C[id_sommet+1];								//GLOBAL
		range[2] = range[1] - range[0];
	}

	volatile __shared__ int comm[NWARPS][3];
	volatile __shared__ int shared_sommet[NWARPS];
	uint32_t sum;

	while( __any(range[2]) )
	{

		int voisin = -1;

		if(range[2])
			comm[warp_id][0] = lane_id;							// SHARED

		if(comm[warp_id][0] == lane_id)
		{
			comm[warp_id][1] = range[0];							// SHARED
			comm[warp_id][2] = range[1];							// SHARED
			range[2] = 0;
			shared_sommet[warp_id] = id_sommet;					// SHARED
		}

		int r_gather = comm[warp_id][1] + lane_id;
		int r_gather_end = comm[warp_id][2];
		while(r_gather < r_gather_end)
		{
			sum = 0;
			voisin = R[r_gather];								// GLOBAL

			unsigned int position = voisin / BITMAP_UNIT_SIZE;
			BITMAP_UNIT mask = tex1Dfetch(tex_visited,position);
			BITMAP_UNIT mask_bit = 1 << (voisin % BITMAP_UNIT_SIZE);
			if(!(mask & mask_bit))
			{
				visited_tex[position] |= mask_bit;
				//int32_t value = atomicCAS(&visited_label[voisin],-1,shared_sommet[warp_id]+d_offset);
				if(visited_label[voisin] == -1)
					visited_label[voisin] = shared_sommet[warp_id]+d_offset;

				if(visited_label[voisin] == shared_sommet[warp_id]+d_offset)
				{
					unsigned int val_out_queue = 1 << voisin%32;  
					atomicOr(&out_queue[voisin/32],val_out_queue);
					sum = 1;
				}
			}

			// TODO faire à la fin 
			if(__any(sum))
			{
				sum = warpReduceSum(sum);
				if(lane_id == 0)
					atomicAdd(n_out_queue,sum);
			}

			r_gather+=32;
		}

	}
}

__global__ void exscan_maj_kernel(BITMAP_UNIT * d_in_queue, BITMAP_UNIT * d_prefix)
{
	unsigned int id = threadIdx.x+blockDim.x*blockIdx.x;
	d_in_queue[id] |= d_prefix[id];
}

__global__ void exscan_maj_assigned_kernel(BITMAP_UNIT * d_assigned, BITMAP_UNIT * d_out_queue,BITMAP_UNIT * d_prefix)
{
	unsigned int id = threadIdx.x+blockDim.x*blockIdx.x;
	d_assigned[id] |= (d_out_queue[id] & d_prefix[id]) ^ d_out_queue[id];
}			

__global__ void bcast_line_maj_out_queue_kernel(BITMAP_UNIT * d_out_queue,BITMAP_UNIT * d_prefix)
{
	unsigned int id = threadIdx.x+blockDim.x*blockIdx.x;
	d_out_queue[id] |= d_prefix[id];	
}

__global__ void bcast_line_maj_visited_label_kernel(BITMAP_UNIT * out_queue, int32_t * visited_label, BITMAP_UNIT * visited_tex)
{

	unsigned int word = threadIdx.x+blockIdx.x*blockDim.x;
	unsigned int val = 1;
	uint32_t val_out_queue = out_queue[word];
	atomicOr(&visited_tex[word],val_out_queue);

	if(val_out_queue == 0)
		return;

	#pragma unroll
	for(unsigned int i = 0 ; i < 32 ; ++i)
	{
		if((val & val_out_queue) && visited_label[word*32+i] == -1)
		{
			visited_label[word*32+i] = -2;
		}
		val <<= 1;
	}
}

__global__ void pred_maj_visited_label_kernel(BITMAP_UNIT * assigned, int32_t * visited_label)
{
	unsigned int i = threadIdx.x + blockDim.x * blockIdx.x;
	for(unsigned int j = 0 ; j < BITMAP_UNIT_SIZE ; ++j)
	{
		if((visited_label[i*BITMAP_UNIT_SIZE+j] != -1) && !(assigned[i] & 1 << j))
		{
			visited_label[i*BITMAP_UNIT_SIZE+j] = -1;
		}
		if(visited_label[i*BITMAP_UNIT_SIZE+j] == -2)
		{
			visited_label[i*BITMAP_UNIT_SIZE+j] = -1;
		}
	}
}

__global__ void pred_compute_recup_pred(int32_t * visited, int64_t * pred)
{
	int thid = threadIdx.x + blockDim.x * blockIdx.x;
	if(visited[thid] != -1 && pred[thid%d_nvertices_1d_per_proc] == -1)
	{			
		pred[thid%d_nvertices_1d_per_proc] = (int64_t)visited[thid];
	}
}

void run_bfs(int64_t root, int64_t* pred)
{

	#ifdef BFS_DATA
	if(bfs_rank == 0) printf("\n### BFS ###\nlvl   nvertices \n");
	double begin_time_start = MPI_Wtime();
	#endif

	unsigned int curlevel = 1;

	#ifdef BFS_DATA
	double steps_time[7] = {0.f,0.f,0.f,0.f,0.f,0.f,0.f};
	double sub_step[5] = {0.f,0.f,0.f,0.f,0.f};
	#endif

	// Démarrer les files sur les GPU
	HANDLE_ERROR(cudaMemset(d_in_queue,0,sizeof(BITMAP_UNIT)*nwords_per_proc_line(graph.lg_nvertices)));
	HANDLE_ERROR(cudaMemset(d_out_queue,0,sizeof(BITMAP_UNIT)*nwords_per_proc_line(graph.lg_nvertices)));
	HANDLE_ERROR(cudaMemset(d_assigned,0,sizeof(BITMAP_UNIT)*nwords_per_proc_line(graph.lg_nvertices)));
	HANDLE_ERROR(cudaMemset(d_prefix,0,sizeof(BITMAP_UNIT)*nwords_per_proc_line(graph.lg_nvertices)));
	HANDLE_ERROR(cudaMemset(d_visited_tex,0,sizeof(char)*(nvertices_2d_per_proc(graph.lg_nvertices)/8)));

	raz_visited_pred_kernel<<< nvertices_2d_per_proc(graph.lg_nvertices) / 32 , 32 >>>(d_visited_label,d_pred);
	setup_bfs_kernel<<< 1 , 1 >>>(root,d_in_queue,d_visited_label,d_visited_tex,d_pred);

	//transfert_visited_label_to_device(graph.visited_label);	

	#ifdef BFS_DATA
	double begin_time_stop = MPI_Wtime();
	steps_time[0] = begin_time_stop - begin_time_start;
	#endif
	
	graph.n_out_queue = 1;
	
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
			//printf("Nouvelle in_queue : %d\n",graph.n_out_queue);
			// Définition de la borne
			//if(graph.n_out_queue < nvertices_2d_per_proc(graph.lg_nvertices)*0.75)
			if(curlevel < 3)		
			{
#ifdef BFS_DATA
				if(bfs_rank == 0) printf("CSC ");
#endif
				double debut_GPU = MPI_Wtime();
				// Version GPU
				HANDLE_ERROR(cudaMemset(d_n_out_queue,0,sizeof(uint32_t)));
				HANDLE_ERROR(cudaMemset(d_out_queue,0,sizeof(BITMAP_UNIT)*nwords_per_proc_line(graph.lg_nvertices)));
				HANDLE_ERROR(cudaBindTexture(0, tex_visited, d_visited_tex,sizeof(BITMAP_UNIT)*nwords_per_proc_line(graph.lg_nvertices)));
				cudaDeviceSynchronize();
				
				explore_frontier_CSC_warps_v2<<< nwords_per_proc_line(graph.lg_nvertices)/NWARPS , 32*NWARPS >>>( d_in_queue, d_out_queue, d_visited_label,d_visited_tex, d_n_out_queue, d_CSC_R, d_CSC_C);
				HANDLE_ERROR(cudaMemcpy(&graph.n_out_queue,d_n_out_queue,sizeof(uint32_t),cudaMemcpyDeviceToHost));
				HANDLE_ERROR(cudaUnbindTexture(tex_visited));
#ifdef BFS_DATA				
if(bfs_rank==0) printf("= %.8f\n",MPI_Wtime()-debut_GPU);
#endif
			}else{
#ifdef BFS_DATA
				if(bfs_rank == 0) printf("CSR ");
#endif
				double debut_GPU = MPI_Wtime();
				// Version GPU
				HANDLE_ERROR(cudaMemset(d_n_out_queue,0,sizeof(uint32_t)));
				HANDLE_ERROR(cudaMemset(d_out_queue,0,sizeof(BITMAP_UNIT)*nwords_per_proc_line(graph.lg_nvertices)));
				HANDLE_ERROR(cudaBindTexture(0, tex_in_queue, d_in_queue,sizeof(BITMAP_UNIT)*nwords_per_proc_line(graph.lg_nvertices)));
				cudaDeviceSynchronize();
				
				explore_frontier_CSR_warps_v2<<< nwords_per_proc_line(graph.lg_nvertices)/NWARPS , 32*NWARPS >>>(d_out_queue, d_visited_label,d_visited_tex, d_n_out_queue, d_CSR_R, d_CSR_C);
				HANDLE_ERROR(cudaMemcpy(&graph.n_out_queue,d_n_out_queue,sizeof(uint32_t),cudaMemcpyDeviceToHost));
				HANDLE_ERROR(cudaUnbindTexture(tex_in_queue));
#ifdef BFS_DATA
				if(bfs_rank == 0) printf("= %.8f\n",MPI_Wtime()-debut_GPU);
#endif
			}
			
			cudaDeviceSynchronize();
		}

		#ifdef BFS_DATA
		double step_1_stop = MPI_Wtime();
		steps_time[1] += step_1_stop - step_1_start;
		//if(bfs_rank==0) printf("Step 1: %.4f\n",step_1_stop-step_1_start);
		#endif

		#ifdef BFS_DATA
		double step_2_start = MPI_Wtime();
		#endif
		// Vérification de l'arrêt sur le CPU
		{
			uint32_t total;
			MPI_Allreduce(&(graph.n_out_queue),&total,1,MPI_UINT32_T,MPI_SUM,MPI_COMM_WORLD);
			graph.n_out_queue = total;
			#ifdef BFS_DATA
			if(bfs_rank==0) printf("%d     %u \n",curlevel,graph.n_out_queue);
			#endif
			if(graph.n_out_queue==0)
			{
#ifdef BFS_DATA
				if(bfs_rank == 0) printf("bfs end level %d\n",curlevel);
#endif
				break;
			}
		}
		#ifdef BFS_DATA
		double step_2_stop = MPI_Wtime();
		steps_time[2] += step_2_stop - step_2_start;
		//if(bfs_rank==0) printf("Step 1: %.4f\n",step_1_stop-step_1_start);
		#endif

		// étape 2, échange en ligne 
		#ifdef BFS_DATA 
		double step_3_start = MPI_Wtime();
		#endif
		{
			#if PROCESS_COUNT==1
			HANDLE_ERROR(cudaMemset(d_prefix,0,sizeof(BITMAP_UNIT)*nwords_per_proc_line(graph.lg_nvertices)));
			#else
			//	MPI_Exscan(d_out_queue,d_prefix,nwords_per_proc_line(graph.lg_nvertices),BITMAP_UNIT_MPI,MPI_BOR,mpi_line_comm);
			//#endif
			//#ifdef COMM_V2
			// test en utilisant le CPU 
			HANDLE_ERROR(cudaMemcpy(d_in_queue,d_out_queue,nwords_per_proc_line(graph.lg_nvertices)*sizeof(BITMAP_UNIT),cudaMemcpyDeviceToDevice));
			cudaDeviceSynchronize();
			MPI_Status stat;
			if(line_rank != 0)
			{
				MPI_Recv(d_prefix,nwords_per_proc_line(graph.lg_nvertices),BITMAP_UNIT_MPI,line_rank-1,0,mpi_line_comm,&stat);
				// Kernel de maj 
				cudaDeviceSynchronize();
				exscan_maj_kernel<<< nwords_per_proc_line(graph.lg_nvertices) / (NWARPS*32) , NWARPS*32 >>>(d_in_queue,d_prefix);
				cudaDeviceSynchronize();

			}else{
				MPI_Send(d_in_queue,nwords_per_proc_line(graph.lg_nvertices),BITMAP_UNIT_MPI,line_rank+1,0,mpi_line_comm);
				cudaDeviceSynchronize();
				HANDLE_ERROR(cudaMemset(d_prefix,0,sizeof(BITMAP_UNIT)*nwords_per_proc_line(graph.lg_nvertices)));
				cudaDeviceSynchronize();
			}

			if(line_rank != 0 && line_rank != line_size-1)
			{
				MPI_Send(d_in_queue,nwords_per_proc_line(graph.lg_nvertices),BITMAP_UNIT_MPI,line_rank+1,0,mpi_line_comm);
				cudaDeviceSynchronize();
			}
			#endif
			
			exscan_maj_assigned_kernel<<<nwords_per_proc_line(graph.lg_nvertices)/(32*NWARPS) , 32*NWARPS >>>(d_assigned, d_out_queue, d_prefix);
			cudaDeviceSynchronize();
		}
		#ifdef BFS_DATA
		double step_3_stop = MPI_Wtime();
		steps_time[3] += step_3_stop - step_3_start;
		//if(bfs_rank==0) printf("Step 2: %.4f\n",step_2_stop-step_2_start);
		#endif

		// étape 3, échange outQ
		// Le proc en bout de ligne génère la outQ générale 
		#ifdef BFS_DATA
		double step_4_start = MPI_Wtime();
		#endif
		{
			#ifdef BFS_DATA 
			double sub_step_1_4_start = MPI_Wtime();
			#endif

			if(col_num == (line_size-1))
				bcast_line_maj_out_queue_kernel<<< nwords_per_proc_line(graph.lg_nvertices) / (NWARPS*32) , NWARPS*32 >>>(d_out_queue, d_prefix);

			cudaDeviceSynchronize();

			MPI_Bcast(d_out_queue,nwords_per_proc_line(graph.lg_nvertices),BITMAP_UNIT_MPI,line_size-1,mpi_line_comm);
			// Test du bcast à la suite 
			#ifdef COMM_V2
			MPI_Status stat;
			if(line_rank != line_size-1)
			{
				MPI_Recv(d_out_queue,nwords_per_proc_line(graph.lg_nvertices),BITMAP_UNIT_MPI,line_rank+1,0,mpi_line_comm,&stat);
				// Kernel de maj 
				cudaDeviceSynchronize();
			}else{
				MPI_Send(d_out_queue,nwords_per_proc_line(graph.lg_nvertices),BITMAP_UNIT_MPI,line_rank-1,0,mpi_line_comm);
				cudaDeviceSynchronize();
			}

			if(line_rank != 0 && line_rank != line_size-1)
			{
				MPI_Send(d_out_queue,nwords_per_proc_line(graph.lg_nvertices),BITMAP_UNIT_MPI,line_rank-1,0,mpi_line_comm);
				cudaDeviceSynchronize();
			}
			cudaDeviceSynchronize();
			#endif

			#ifdef BFS_DATA
			double sub_step_1_4_stop = MPI_Wtime();
			sub_step[0] += sub_step_1_4_stop - sub_step_1_4_start;
			#endif

			/* Mise à jour des visites */
			//transfert_out_queue_to_device(graph.out_queue);
			bcast_line_maj_visited_label_kernel<<< nwords_per_proc_line(graph.lg_nvertices) / (NWARPS*32) , NWARPS*32>>>(d_out_queue,d_visited_label, d_visited_tex);
			cudaDeviceSynchronize();

			#ifdef BFS_DATA
			double sub_step_2_4_stop = MPI_Wtime();
			sub_step[1] += sub_step_2_4_stop - sub_step_1_4_stop;
			#endif

		}
		#ifdef BFS_DATA
		double step_4_stop = MPI_Wtime();
		steps_time[4] += step_4_stop - step_4_start;
		//if(bfs_rank==0) printf("Step 3: %.4f\n",step_3_stop-step_3_start);
		#endif

		// étape 5, échange inQ et outQ
		#ifdef BFS_DATA
		double step_5_start = MPI_Wtime();
		#endif
		{
			HANDLE_ERROR(cudaMemcpy(d_in_queue,d_out_queue,sizeof(BITMAP_UNIT)*nwords_per_proc_line(graph.lg_nvertices),cudaMemcpyDeviceToDevice));
			cudaDeviceSynchronize();
			MPI_Bcast(d_in_queue,nwords_per_proc_line(graph.lg_nvertices),BITMAP_UNIT_MPI,col_num,mpi_col_comm);
			cudaDeviceSynchronize();
		}

		#ifdef BFS_DATA
		double step_5_stop = MPI_Wtime();
		steps_time[5] += step_5_stop - step_5_start;
		//if(bfs_rank==0) printf("Step 5: %.4f\n",step_5_stop-step_5_start);
		#endif
		++curlevel;
	}


	#ifdef BFS_DATA
	double step_pred_start = MPI_Wtime();
	#endif
	
	// maj pred GPU en fonction des assigned 
	pred_maj_visited_label_kernel<<< nwords_per_proc_line(graph.lg_nvertices) / (NWARPS*32) , NWARPS*32 >>>(d_assigned,d_visited_label);
	cudaDeviceSynchronize();

	//HANDLE_ERROR(cudaMemcpy(graph.visited_label,d_visited_label,sizeof(int32_t)*nvertices_2d_per_proc(graph.lg_nvertices),cudaMemcpyDeviceToHost));

	#ifdef BFS_DATA
	double sub_step_1_pred_stop = MPI_Wtime();
	sub_step[2] = sub_step_1_pred_stop - step_pred_start;
	#endif

	// Echanger tableaux sur la ligne 
	#if PROCESS_COUNT!=1
	MPI_Alltoall(d_visited_label,nvertices_1d_per_proc(graph.lg_nvertices),MPI_INT,d_visited_label_tmp,nvertices_1d_per_proc(graph.lg_nvertices),MPI_INT,mpi_line_comm);
	cudaDeviceSynchronize();
	#else 
	HANDLE_ERROR(cudaMemcpy(d_visited_label_tmp,d_visited_label,sizeof(int32_t)*nvertices_2d_per_proc(graph.lg_nvertices),cudaMemcpyDeviceToDevice));
	#endif

	#ifdef BFS_DATA
	double sub_step_2_pred_stop = MPI_Wtime();
	sub_step[3] = sub_step_2_pred_stop - sub_step_1_pred_stop;
	#endif

	// Calcul et récupération de pred
	pred_compute_recup_pred<<< nvertices_2d_per_proc(graph.lg_nvertices) / (NWARPS*32) , NWARPS*32 >>>(d_visited_label_tmp, d_pred);
	HANDLE_ERROR(cudaMemcpy(pred,d_pred,sizeof(int64_t)*nvertices_1d_per_proc(graph.lg_nvertices),cudaMemcpyDeviceToHost));

	#ifdef BFS_DATA
	double step_pred_stop = MPI_Wtime();
	sub_step[4] = step_pred_stop - sub_step_2_pred_stop;
	steps_time[6] += step_pred_stop - step_pred_start;
	//if(bfs_rank==0) printf("Step pred: %.4f\n",step_pred_stop-step_pred_start);
	#endif

	#ifdef BFS_DATA
	if(bfs_rank==0)
	{
		double sum = 0;
		double median_time[5] = {0.f,0.f,0.f,0.f,0.f};
		for(unsigned int i = 0 ; i < 7 ; ++i)
		{
				if(i > 0 && i < 6)
					median_time[i%5] = steps_time[i] / curlevel;
				sum += steps_time[i];
		}
		printf("step         time    average  percent   \n");
		for(unsigned int i = 0 ; i < 7 ; ++i)
		{
				switch(i)
				{
					case 0 : printf("Setup       %2.4f          %2.4f \n",steps_time[i],steps_time[i]*100/sum); break;
					case 1 : printf("Exploration %2.4f  %2.4f  %2.4f \n",steps_time[i],median_time[i%5],steps_time[i]*100/sum); break;
					case 2 : printf("AllReduce   %2.4f  %2.4f  %2.4f \n",steps_time[i],median_time[i%5],steps_time[i]*100/sum); break;
					case 3 : printf("Exscan      %2.4f  %2.4f  %2.4f \n",steps_time[i],median_time[i%5],steps_time[i]*100/sum); break;
					case 4 : 
						printf("BCast line  %2.4f  %2.4f  %2.4f \n",steps_time[i],median_time[i%5],steps_time[i]*100/sum); 
						printf("     BCast  %2.4f \n",sub_step[0]);
						printf("     Calc   %2.4f \n",sub_step[1]);
						break;
					case 5 : printf("BCast col   %2.4f  %2.4f  %2.4f \n",steps_time[i],median_time[i%5],steps_time[i]*100/sum); break;
					case 6 : 
						printf("Pred        %2.4f          %2.4f \n",steps_time[i],steps_time[i]*100/sum); 
						printf("     Calc   %2.4f \n",sub_step[2]);
						printf("     Atoa   %2.4f \n",sub_step[3]);
						printf("     Calc   %2.4f \n",sub_step[4]);
						break;
				}
		}
	}
	fflush(stdout);
	#endif
	//exit(0);
}

