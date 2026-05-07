# ============================================================================
# Prompt Evolution Module
# ============================================================================
import os
import os.path as osp
import sys
import json
import yaml
import shutil
import asyncio
import logging
import pickle
import glob
import numpy as np
import networkx as nx
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Any, Optional, List, Iterable, Tuple
from chromadb import PersistentClient as ChromaClient
from chromadb.config import Settings
from chromadb.utils import embedding_functions
from pyclustering.cluster.bang import bang
from internagent.mas.agents.agent_factory import AgentFactory
from internagent.mas.models.model_factory import ModelFactory
logger = logging.getLogger(__name__)
CHROMA_AVAILABLE = True

@dataclass
class IdeaGraph:
    """
    Graph-based idea storage and retrieval system.

    Stores ideas in a graph structure where nodes are ideas and edges represent
    similarity relationships. Uses ChromaDB for vector storage and NetworkX for
    graph operations.

    Attributes:
        working_dir (str): Directory for storing graph data
        namespace (str): Namespace identifier for the graph
        similarity_threshold (float): Minimum similarity for creating edges (default: 0.7)
    """

    working_dir: str
    namespace: str
    similarity_threshold: float = 0.7

    def __post_init__(self):
        """Initialize the idea graph and load existing data if available."""
        
        # Initialize ChromaDB
        os.makedirs(self.working_dir, exist_ok=True)
        chroma_db_path = osp.join(self.working_dir, "chroma_db")
        
        # 修改这里：使用新的 PersistentClient 替代旧的 Client + Settings
        self.chroma_client = ChromaClient(path=chroma_db_path)
        
        # Use OpenAI embeddings (can be configured)
        self.embedding_function = embedding_functions.OpenAIEmbeddingFunction(
            api_key=os.environ.get("OPENAI_API_KEY", ""),
            api_base=os.environ.get("OPENAI_API_BASE_URL", ""),
            model_name="text-embedding-3-small"
        )
        
        # Get or create collection
        try:
            self.collection = self.chroma_client.get_or_create_collection(
                name=self.namespace,
                embedding_function=self.embedding_function
            )
        except Exception as e:
            logger.error(f"Failed to create ChromaDB collection: {e}")
            self.collection = None
        
        # Define paths
        self._graph_pic_save_path = osp.join(self.working_dir, 'idea_graph.png')
        self._node_match_save_path = osp.join(self.working_dir, 'match_nodes.txt')
        self._graph_save_path = osp.join(self.working_dir, f'{self.namespace}_graph.pkl')
        
        # Load or create graph
        if osp.exists(self._graph_save_path):
            try:
                with open(self._graph_save_path, 'rb') as f:
                    self.graph = pickle.load(f)
                logger.info(f"IdeaGraph loaded from {self._graph_save_path}")
            except Exception as e:
                logger.error(f"Failed to load graph: {e}")
                self.graph = nx.Graph()
        else:
            self.graph = nx.Graph()
            logger.info("New empty IdeaGraph created")


    def add_idea_node(self, idea: Dict[str, Any]) -> None:
        """
        Add an idea node to the graph and connect it to similar ideas.

        Args:
            idea (Dict[str, Any]): Idea dictionary containing:
                - id (str): Unique identifier
                - name (str): Idea name/title
                - description (str): Detailed description
                - Other metadata fields
        """
        if not CHROMA_AVAILABLE or self.collection is None:
            logger.warning("ChromaDB not available, skipping idea addition")
            return

        idea_id = idea.get('id', idea.get('name', str(len(self.graph.nodes))))
        idea_name = idea.get('name', '')
        idea_description = idea.get('description', '')

        # Create a searchable text representation
        idea_text = f"{idea_name}: {idea_description}"

        # Check if node already exists
        if idea_id in self.graph:
            logger.info(f"Idea node {idea_id} already exists, skipping")
            return

        # Add idea to vector store
        try:
            self.collection.add(
                documents=[idea_text],
                ids=[idea_id],
                metadatas=[{
                    "name": idea_name,
                    "description": idea_description,
                    "id": idea_id
                }]
            )
        except Exception as e:
            logger.error(f"Failed to add idea to ChromaDB: {e}")
            return

        # Add node to graph
        self.graph.add_node(idea_id, **idea)

        # Find similar ideas and create edges
        try:
            results = self.collection.query(
                query_texts=[idea_text],
                n_results=min(10, len(self.graph.nodes))
            )

            if results and results['ids'] and len(results['ids']) > 0:
                neighbor_ids = results['ids'][0]
                distances = results['distances'][0]

                for neighbor_id, distance in zip(neighbor_ids, distances):
                    # Skip self
                    if neighbor_id == idea_id:
                        continue

                    # Convert distance to similarity (assuming cosine distance)
                    similarity = 1 - distance

                    if similarity < self.similarity_threshold:
                        continue

                    # Ensure neighbor exists in graph
                    if neighbor_id not in self.graph:
                        logger.warning(f"Neighbor {neighbor_id} not in graph, skipping edge")
                        continue

                    # Add edge with similarity weight
                    self.graph.add_edge(idea_id, neighbor_id, weight=similarity)
                    logger.debug(f"Added edge: {idea_id} <-> {neighbor_id} (similarity: {similarity:.3f})")

        except Exception as e:
            logger.error(f"Failed to find similar ideas: {e}")

        # Save graph
        self._save_graph()
        logger.info(f"Added idea node: {idea_id} ({idea_name})")

    def add_ideas_batch(self, ideas: List[Dict[str, Any]]) -> None:
        """
        Add multiple ideas to the graph in batch.

        Args:
            ideas (List[Dict[str, Any]]): List of idea dictionaries
        """
        logger.info(f"Adding {len(ideas)} ideas to graph in batch")
        for idea in ideas:
            self.add_idea_node(idea)

    def retrieve_related_ideas(
        self,
        query_idea: str,
        node_num: int = 5,
        hop: int = 1
    ) -> List[str]:
        """
        Retrieve related ideas from the graph based on similarity and neighborhood expansion.

        Args:
            query_idea (str): The idea used as the query input
            node_num (int): Number of top similar ideas to retrieve
            hop (int): Number of hops for neighborhood expansion (default: 1)

        Returns:
            List[str]: List of related idea node IDs
        """
        if not CHROMA_AVAILABLE or self.collection is None:
            logger.warning("ChromaDB not available, returning empty list")
            return []

        try:
            # Query similar ideas
            results = self.collection.query(
                query_texts=[query_idea],
                n_results=node_num
            )

            if not results or not results['ids'] or len(results['ids']) == 0:
                return []

            top_nodes = results['ids'][0]

            # Expand neighborhood
            related_nodes = set(top_nodes)
            for node_id in top_nodes:
                if node_id in self.graph:
                    try:
                        neighbors = nx.single_source_shortest_path_length(
                            self.graph, node_id, cutoff=hop
                        ).keys()
                        related_nodes.update(neighbors)
                    except nx.NetworkXError as e:
                        logger.warning(f"Failed to expand neighbors for {node_id}: {e}")

            return list(related_nodes)

        except Exception as e:
            logger.error(f"Failed to retrieve related ideas: {e}")
            return []

    def cluster_ideas(self, method: str = "louvain") -> None:
        """
        Perform clustering on ideas in the graph and assign cluster IDs.

        Args:
            method (str): Clustering method to use. Options:
                - "louvain": Community detection using Louvain algorithm (default)
                - "spectral": Spectral clustering
                - "embedding": Clustering based on embeddings (requires FINCH)
        """
        nodes = list(self.graph.nodes)

        if len(nodes) == 0:
            logger.warning("No nodes in graph, skipping clustering")
            return

        logger.info(f"Clustering {len(nodes)} ideas using method: {method}")

        if method == "louvain":
            # Use Louvain community detection
            try:
                communities = nx.community.louvain_communities(self.graph, seed=42)
                for cluster_id, community in enumerate(communities):
                    for node_id in community:
                        self.graph.nodes[node_id]['cluster_id'] = cluster_id
                logger.info(f"Louvain clustering created {len(communities)} clusters")
            except Exception as e:
                logger.error(f"Louvain clustering failed: {e}")
                # Fallback: assign all to cluster 0
                for node_id in nodes:
                    self.graph.nodes[node_id]['cluster_id'] = 0

        elif method == "spectral":
            # Use spectral clustering (requires connected components)
            try:
                # Get largest connected component
                if nx.is_connected(self.graph):
                    adj_matrix = nx.to_numpy_array(self.graph)
                    from sklearn.cluster import SpectralClustering
                    n_clusters = min(5, len(nodes))
                    clustering = SpectralClustering(n_clusters=n_clusters, affinity='precomputed')
                    labels = clustering.fit_predict(adj_matrix)
                    for node_id, label in zip(nodes, labels):
                        self.graph.nodes[node_id]['cluster_id'] = int(label)
                    logger.info(f"Spectral clustering created {n_clusters} clusters")
                else:
                    logger.warning("Graph not connected, using component-based clustering")
                    for cluster_id, component in enumerate(nx.connected_components(self.graph)):
                        for node_id in component:
                            self.graph.nodes[node_id]['cluster_id'] = cluster_id
            except Exception as e:
                logger.error(f"Spectral clustering failed: {e}")
                for node_id in nodes:
                    self.graph.nodes[node_id]['cluster_id'] = 0

        elif method == "embedding":
            # Use embedding-based clustering with FINCH or K-means
            if not CHROMA_AVAILABLE or self.collection is None:
                logger.error("ChromaDB not available for embedding clustering")
                return

            try:
                embeddings = []
                valid_nodes = []

                for node_id in nodes:
                    # Get embedding from ChromaDB
                    result = self.collection.get(ids=[node_id], include=['embeddings'])
                    if result and result['embeddings'] and len(result['embeddings']) > 0:
                        embeddings.append(result['embeddings'][0])
                        valid_nodes.append(node_id)

                if len(embeddings) == 0:
                    logger.warning("No embeddings found")
                    return

                X = np.vstack(embeddings)

                # Use simple K-means clustering as fallback
                from sklearn.cluster import KMeans
                n_clusters = min(5, len(valid_nodes))
                kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                labels = kmeans.fit_predict(X)

                for node_id, label in zip(valid_nodes, labels):
                    self.graph.nodes[node_id]['cluster_id'] = int(label)

                logger.info(f"Embedding clustering created {n_clusters} clusters")

            except Exception as e:
                logger.error(f"Embedding clustering failed: {e}")
                for node_id in nodes:
                    self.graph.nodes[node_id]['cluster_id'] = 0

        else:
            logger.error(f"Unknown clustering method: {method}")
            return

        # Save graph with cluster assignments
        self._save_graph()

    def get_cluster_summary(self) -> Dict[int, List[str]]:
        """
        Get a summary of clusters and their member ideas.

        Returns:
            Dict[int, List[str]]: Dictionary mapping cluster_id to list of idea IDs
        """
        clusters = {}
        for node_id in self.graph.nodes:
            cluster_id = self.graph.nodes[node_id].get('cluster_id', 0)
            if cluster_id not in clusters:
                clusters[cluster_id] = []
            clusters[cluster_id].append(node_id)
        return clusters

    def _save_graph(self) -> None:
        """Save the graph to disk."""
        try:
            with open(self._graph_save_path, "wb") as f:
                pickle.dump(self.graph, f)
            logger.debug(f"Graph saved to {self._graph_save_path}")
        except Exception as e:
            logger.error(f"Failed to save graph: {e}")

    def __iter__(self) -> Iterable[Tuple[str, int]]:
        """Iterate over nodes and their cluster IDs."""
        return ((node, self.graph.nodes[node].get('cluster_id', 0)) for node in self.graph.nodes)

    def get_graph_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the idea graph.

        Returns:
            Dict containing graph statistics
        """
        return {
            "num_nodes": len(self.graph.nodes),
            "num_edges": len(self.graph.edges),
            "density": nx.density(self.graph) if len(self.graph.nodes) > 0 else 0,
            "is_connected": nx.is_connected(self.graph) if len(self.graph.nodes) > 0 else False,
            "num_components": nx.number_connected_components(self.graph) if len(self.graph.nodes) > 0 else 0,
            "avg_degree": sum(dict(self.graph.degree()).values()) / len(self.graph.nodes) if len(self.graph.nodes) > 0 else 0
        }


class PromptEvolver:
    """
    Handles evolution of research prompts based on accumulated experiences.

    This class analyzes positive and negative experience libraries to:
    1. Identify the best performing method based on metrics
    2. Generate new task directions building on successful patterns
    3. Update background information with insights from best methods
    """

    def __init__(self, args, logger, config=None, idea_graph=None):
        """
        Initialize the prompt evolver.

        Args:
            args: Command-line arguments
            logger: Logger instance
            config: Configuration dictionary
            idea_graph: Optional IdeaGraph instance for exploratory prompt selection
        """
        self.args = args
        self.logger = logger
        self.config = config or {}
        self.prompt_agent = None
        self.model_factory = ModelFactory()
        self.agent_factory = AgentFactory()
        self.idea_graph = idea_graph

        # Number of parallel prompts to generate for exploration
        self.num_candidates = config.get("agents", {}).get("prompt_evolver", {}).get("num_candidates", 3)

    def _initialize_agent(self):
        """Initialize the PromptGeneratorAgent using AgentFactory."""
        if self.prompt_agent is not None:
            return
            # Create agents dict similar to interface.py
        self.prompt_agent = self.agent_factory.create_agent(
            agent_type="prompt_evolver",
            config=self.config,
            model_factory=self.model_factory
        )
        self.logger.info(f"Prompt Agent initialized via AgentFactory")
            # Get the prompt_evolver agent

    async def _generate_single_prompt_candidate(
        self,
        library_formatted: Dict[str, Any],
        current_task: str,
        current_background: str,
        task_domain: str,
        candidate_id: int,
        fix_direction: str,
    ) -> Dict[str, Any]:
        """
        Generate a single prompt candidate.

        Args:
            library_formatted: Formatted experience library
            current_task: Current task description
            current_background: Current background
            task_domain: Research domain
            candidate_id: ID for this candidate

        Returns:
            Dict containing the generated prompt and metadata
        """
        try:
            result = await self.prompt_agent.execute(
                context={
                    "experience_library": library_formatted,
                    "current_task": current_task,
                    "current_background": current_background,
                    "domain": task_domain,
                    "fix_direction" : fix_direction
                },
                params={
                    "generate_task": self.config.get("agents", {}).get("prompt_evolver", {}).get("generate_task", True),
                    "generate_background": self.config.get("agents", {}).get("prompt_evolver", {}).get("generate_background", False)
                }
            )

            return {
                "candidate_id": candidate_id,
                "new_task": result["new_task"],
                "new_background": result.get("new_background", current_background),
                "experiences_used": result.get("experiences_used", 0),
                "success": True
            }
        except Exception as e:
            self.logger.error(f"Failed to generate candidate {candidate_id}: {e}")
            return {
                "candidate_id": candidate_id,
                "success": False,
                "error": str(e)
            }

    def _calculate_exploration_score(self, prompt_candidate: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
        """
        Calculate exploration score for a prompt candidate based on IdeaGraph.

        Lower score = more exploratory (less similar to existing ideas).

        Args:
            prompt_candidate: Candidate prompt dictionary with "new_task"

        Returns:
            Tuple of (exploration_score, metadata)
        """
        if self.idea_graph is None or not CHROMA_AVAILABLE:
            self.logger.warning("IdeaGraph not available, returning random score")
            return (0.5, {"reason": "no_graph"})

        new_task = prompt_candidate.get("new_task", "")
        if not new_task:
            return (1.0, {"reason": "empty_task"})

        try:
            # Retrieve related ideas from graph
            related_idea_ids = self.idea_graph.retrieve_related_ideas(
                query_idea=new_task,
                node_num=10,
                hop=1
            )

            if not related_idea_ids:
                # No similar ideas found - highly exploratory!
                self.logger.info(f"Candidate {prompt_candidate['candidate_id']}: No similar ideas found (highly exploratory)")
                return (0.0, {
                    "reason": "no_similar_ideas",
                    "related_count": 0,
                    "cluster_sizes": {}
                })

            # Count ideas per cluster
            cluster_counts = {}
            for idea_id in related_idea_ids:
                if idea_id in self.idea_graph.graph:
                    cluster_id = self.idea_graph.graph.nodes[idea_id].get('cluster_id', 0)
                    cluster_counts[cluster_id] = cluster_counts.get(cluster_id, 0) + 1

            # Calculate exploration score
            # More related ideas = less exploratory
            # Larger clusters = less exploratory
            total_related = len(related_idea_ids)
            max_cluster_size = max(cluster_counts.values()) if cluster_counts else 0

            # Exploration score: weighted combination of total related and max cluster size
            # Normalize by graph size
            graph_size = len(self.idea_graph.graph.nodes)
            if graph_size > 0:
                exploration_score = (0.6 * (total_related / graph_size)) + (0.4 * (max_cluster_size / graph_size))
            else:
                exploration_score = 0.0

            metadata = {
                "reason": "calculated",
                "related_count": total_related,
                "cluster_sizes": cluster_counts,
                "max_cluster_size": max_cluster_size,
                "graph_size": graph_size
            }

            self.logger.info(
                f"Candidate {prompt_candidate['candidate_id']}: "
                f"Exploration score={exploration_score:.3f}, "
                f"Related ideas={total_related}, "
                f"Max cluster size={max_cluster_size}"
            )

            return (exploration_score, metadata)

        except Exception as e:
            self.logger.error(f"Failed to calculate exploration score: {e}")
            return (0.5, {"reason": "error", "error": str(e)})
        

    async def evolve_prompt(
        self,
        library_path,
        current_prompt_path,
        output_path,
        task_domain="machine learning",
        fix_direction = "",
    ):
        """
        Evolve the prompt.json with updated task and background based on experience library.

        Generates multiple candidate prompts in parallel and selects the most exploratory one
        based on similarity to existing ideas in the IdeaGraph.

        Args:
            library_path: Path to unified experience library JSON
            current_prompt_path: Path to current prompt.json
            output_path: Path to save new prompt.json
            task_domain: Research domain

        Returns:
            Dict containing the new prompt data
        """
        self._initialize_agent()

        # Load experience library
        try:
            with open(library_path, "r", encoding="utf-8") as f:
                library_data = json.load(f)
            experiences = library_data.get('experiences', [])
            self.logger.info(f"Loaded experience library: {len(experiences)} experiences")
        except Exception as e:
            self.logger.error(f"Failed to load experience library: {e}")
            raise ValueError(f"Could not load experience library from {library_path}")

        if not experiences:
            self.logger.error("No experiences found in library")
            raise ValueError("Experience library is empty")

        # Load current prompt
        try:
            with open(current_prompt_path, "r", encoding="utf-8") as f:
                current_prompt = json.load(f)
            current_task = current_prompt.get("task_description", "")
            current_background = current_prompt.get("background", "")
            self.logger.info(f"Loaded current prompt from: {current_prompt_path}")
        except Exception as e:
            self.logger.error(f"Failed to load current prompt: {e}")
            current_task = ""
            current_background = ""
            current_prompt = {}

        # Format experience library for agent
        library_formatted = {
            "experiences": experiences,  # Pass all experiences, agent will filter
            "count": len(experiences),
            "description": "Unified experience library from contrastive learning"
        }

        # Generate multiple prompt candidates in parallel
        try:
            self.logger.info(f"Generating {self.num_candidates} prompt candidates in parallel...")

            async def generate_candidates():
                tasks = [
                    self._generate_single_prompt_candidate(
                        library_formatted=library_formatted,
                        current_task=current_task,
                        current_background=current_background,
                        task_domain=task_domain,
                        candidate_id=i,
                        fix_direction = fix_direction
                    )
                    for i in range(self.num_candidates)
                ]
                return await asyncio.gather(*tasks)

            candidates = await generate_candidates()

            # Filter successful candidates
            successful_candidates = [c for c in candidates if c.get("success", False)]

            if not successful_candidates:
                raise RuntimeError("All candidate generations failed")

            self.logger.info(f"Successfully generated {len(successful_candidates)}/{self.num_candidates} candidates")

            # Filter out candidates with empty new_task
            valid_candidates = [c for c in successful_candidates if c.get("new_task", "").strip()]
            if not valid_candidates:
                self.logger.warning("All candidates have empty task descriptions, keeping original prompt")
                raise RuntimeError("All candidates generated empty task descriptions")

            self.logger.info(f"Valid candidates with non-empty tasks: {len(valid_candidates)}/{len(successful_candidates)}")

            # Calculate exploration scores for each candidate
            self.logger.info("Calculating exploration scores based on IdeaGraph...")
            scored_candidates = []
            for candidate in valid_candidates:
                score, metadata = self._calculate_exploration_score(candidate)
                scored_candidates.append({
                    **candidate,
                    "exploration_score": score,
                    "exploration_metadata": metadata
                })

            # Select the most exploratory candidate (lowest score)
            best_candidate = min(scored_candidates, key=lambda x: x["exploration_score"])

            self.logger.info("=" * 80)
            self.logger.info("CANDIDATE SELECTION RESULTS:")
            self.logger.info("=" * 80)
            for i, candidate in enumerate(sorted(scored_candidates, key=lambda x: x["exploration_score"])):
                prefix = ">>> SELECTED" if candidate == best_candidate else "   "
                self.logger.info(
                    f"{prefix} Candidate {candidate['candidate_id']}: "
                    f"Exploration score={candidate['exploration_score']:.3f}, "
                    f"Related={candidate['exploration_metadata'].get('related_count', 'N/A')}, "
                    f"Max cluster={candidate['exploration_metadata'].get('max_cluster_size', 'N/A')}"
                )
                self.logger.info(f"    Task preview: {candidate['new_task'][:100]}...")
            self.logger.info("=" * 80)

            # Create new prompt dict
            new_prompt = current_prompt.copy()
            new_prompt["task_description"] = best_candidate["new_task"]


            # Add metadata about generation
            new_prompt["_generation_metadata"] = {
                "generated_at": datetime.now().isoformat(),
                "num_candidates_generated": len(successful_candidates),
                "num_valid_candidates": len(valid_candidates),
                "selected_candidate_id": best_candidate["candidate_id"],
                "exploration_score": best_candidate["exploration_score"],
                "exploration_metadata": best_candidate["exploration_metadata"],
                "experiences_used": best_candidate.get("experiences_used", 0),
                "total_experiences": len(experiences)
            }

            # Save new prompt
            os.makedirs(osp.dirname(output_path), exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(new_prompt, f, indent=4, ensure_ascii=False)

            self.logger.info(f"New prompt saved to: {output_path}")
            self.logger.info(
                f"Selected candidate {best_candidate['candidate_id']} with exploration score "
                f"{best_candidate['exploration_score']:.3f}"
            )

            # Save all candidates for analysis
            candidates_path = output_path.replace(".json", "_candidates.json")
            with open(candidates_path, "w", encoding="utf-8") as f:
                json.dump(scored_candidates, f, indent=4, ensure_ascii=False)
            self.logger.info(f"All candidates saved to: {candidates_path}")

            return new_prompt

        except Exception as e:
            self.logger.error(f"Failed to generate new prompt: {str(e)}")
            import traceback
            traceback.print_exc()
            raise


# ============================================================================
# Memory Module
# ============================================================================
class MemoryModule:
    """
    Memory module for storing and retrieving historical information from:
    1. Idea generation outputs (ideas_*.json files)
    2. Experiment execution notes (notes.txt files)
    """

    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialize the memory module

        Args:
            logger: Logger instance for logging operations
        """
        self.logger = logger or logging.getLogger("MemoryModule")
        self.ideas_history: List[Dict] = []
        self.notes_history: List[Dict] = []
        self.memory_data: Dict = {
            'ideas': [],
            'experiments': [],
            'summary': {
                'total_ideas': 0,
                'total_experiments': 0,
                'successful_experiments': 0,
                'failed_experiments': 0
            }
        }

    def load_idea_generation_output(self, idea_file_path: str) -> Dict:
        """
        Load idea generation output from a JSON file

        Args:
            idea_file_path: Path to the ideas JSON file (e.g., results/{task}/ideas_{session_id}.json)

        Returns:
            Dictionary containing the loaded ideas
        """
        try:
            if not osp.exists(idea_file_path):
                self.logger.warning(f"Idea file not found: {idea_file_path}")
                return {}

            # Check for duplicates - skip if already loaded
            abs_path = osp.abspath(idea_file_path)
            for existing in self.memory_data['ideas']:
                if osp.abspath(existing.get('file_path', '')) == abs_path:
                    self.logger.debug(f"Ideas already loaded, skipping: {idea_file_path}")
                    return existing.get('data', {})

            with open(idea_file_path, 'r', encoding='utf-8') as f:
                ideas_data = json.load(f)

            self.logger.info(f"Loaded ideas from: {idea_file_path}")

            # Store in memory
            idea_entry = {
                'file_path': idea_file_path,
                'timestamp': datetime.now().isoformat(),
                'data': ideas_data,
                'count': len(ideas_data) if isinstance(ideas_data, list) else 1
            }
            self.ideas_history.append(idea_entry)
            self.memory_data['ideas'].append(idea_entry)
            self.memory_data['summary']['total_ideas'] += idea_entry['count']

            return ideas_data

        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse JSON from {idea_file_path}: {str(e)}")
            return {}
        except Exception as e:
            self.logger.error(f"Error loading idea file {idea_file_path}: {str(e)}")
            return {}

    def load_experiment_notes(self, notes_file_path, summary_path) -> Dict:
        """
        Load experiment notes from a notes.txt file

        Args:
            notes_file_path: Path to the notes.txt file (e.g., results/{task}/{experiment_folder}/notes.txt)
            summary_path: Path to the experiment_report.txt file (optional)

        Returns:
            Dictionary containing parsed notes information
        """
        try:
            if not osp.exists(notes_file_path):
                self.logger.warning(f"Notes file not found: {notes_file_path}")
                return {}

            # Check for duplicates - skip if already loaded
            abs_path = osp.abspath(notes_file_path)
            for existing in self.memory_data['experiments']:
                if osp.abspath(existing.get('file_path', '')) == abs_path:
                    self.logger.debug(f"Notes already loaded, skipping: {notes_file_path}")
                    return existing.get('data', {})

            with open(notes_file_path, 'r', encoding='utf-8') as f:
                notes_content = f.read()

            # Load summary content if available, otherwise use empty string
            summary_content = ""
            if summary_path and osp.exists(summary_path):
                with open(summary_path, 'r', encoding='utf-8') as f:
                    summary_content = f.read()

            # Parse notes content
            notes_data = self._parse_notes_content(notes_content, summary_content)
            notes_data['file_path'] = notes_file_path


            self.logger.info(f"Loaded notes from: {notes_file_path}")

            # Store in memory
            notes_entry = {
                'file_path': notes_file_path,
                'timestamp': datetime.now().isoformat(),
                'data': notes_data
            }
            self.notes_history.append(notes_entry)
            self.memory_data['experiments'].append(notes_entry)
            self.memory_data['summary']['total_experiments'] += 1

            return notes_data

        except Exception as e:
            self.logger.error(f"Error loading notes file {notes_file_path}: {str(e)}")
            return {}

    def _parse_notes_content(self, content, summary_content) -> Dict:
        """
        Parse the content of a notes.txt file

        Args:
            content: Raw content of the notes file

        Returns:
            Dictionary with parsed fields (name, title, description, raw_content)
        """
        notes_data = {
            'name': '',
            'title': '',
            'description': '',
            'raw_content': summary_content
        }

        lines = content.split('\n')
        for line in lines:
            line_stripped = line.strip()
            if line_stripped.startswith('# Name:'):
                notes_data['name'] = line_stripped.replace('# Name:', '').strip()
            elif line_stripped.startswith('# Title:'):
                notes_data['title'] = line_stripped.replace('# Title:', '').strip()
            elif line_stripped.startswith('# Description:'):
                notes_data['description'] = line_stripped.replace('# Description:', '').strip()

        return notes_data

    def load_all_ideas_from_directory(self, results_dir: str, task_name: Optional[str] = None) -> List[Dict]:
        """
        Load all idea generation outputs from a results directory

        Args:
            results_dir: Base results directory (e.g., 'results/')
            task_name: Optional specific task name to load from

        Returns:
            List of all loaded ideas data
        """
        all_ideas = []

        search_dir = results_dir

        if not osp.exists(search_dir):
            self.logger.warning(f"Results directory not found: {search_dir}")
            return all_ideas

        # Search for all ideas_*.json files
        for root, dirs, files in os.walk(search_dir):
            for file in files:
                if file.startswith('ideas_') and file.endswith('.json'):
                    file_path = osp.join(root, file)
                    ideas_data = self.load_idea_generation_output(file_path)
                    if ideas_data:
                        all_ideas.append(ideas_data)

        self.logger.info(f"Loaded {len(all_ideas)} idea files from {search_dir}")
        return all_ideas

    def load_all_notes_from_directory(self, results_dir: str, task_name: Optional[str] = None) -> List[Dict]:
        """
        Load all experiment notes from a results directory

        Args:
            results_dir: Base results directory (e.g., 'results/')
            task_name: Optional specific task name to load from

        Returns:
            List of all loaded notes data
        """
        all_notes = []

        
        search_dir = results_dir

        if not osp.exists(search_dir):
            self.logger.warning(f"Results directory not found: {search_dir}")
            return all_notes

        # Search for all notes.txt files
        for root, dirs, files in os.walk(search_dir):
            for file in files:
                if file == 'notes.txt':
                    file_path = osp.join(root, file)
                    summary_path = osp.join(root, "experiment_report.txt")
                    notes_data = self.load_experiment_notes(file_path,summary_path)
                    if notes_data:
                        all_notes.append(notes_data)

        self.logger.info(f"Loaded {len(all_notes)} notes files from {search_dir}")
        return all_notes

    def get_memory_summary(self) -> Dict:
        """
        Get a summary of all loaded memory data

        Returns:
            Dictionary containing memory summary statistics
        """
        return self.memory_data['summary']

    def get_all_ideas(self) -> List[Dict]:
        """
        Get all loaded ideas

        Returns:
            List of all idea entries
        """
        return self.memory_data['ideas']

    def get_all_experiments(self) -> List[Dict]:
        """
        Get all loaded experiment notes

        Returns:
            List of all experiment note entries
        """
        return self.memory_data['experiments']

    def export_memory_to_file(self, output_path: str) -> bool:
        """
        Export all memory data to a JSON file

        Args:
            output_path: Path to save the memory data

        Returns:
            True if successful, False otherwise
        """
        try:
            os.makedirs(osp.dirname(output_path), exist_ok=True)

            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(self.memory_data, f, indent=4, ensure_ascii=False)

            self.logger.info(f"Memory data exported to: {output_path}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to export memory data: {str(e)}")
            return False

    def load_memory_from_file(self, input_path: str) -> bool:
        """
        Load memory data from a previously exported JSON file

        Args:
            input_path: Path to the memory data file

        Returns:
            True if successful, False otherwise
        """
        try:
            if not osp.exists(input_path):
                self.logger.warning(f"Memory file not found: {input_path}")
                return False

            with open(input_path, 'r', encoding='utf-8') as f:
                self.memory_data = json.load(f)

            self.logger.info(f"Memory data loaded from: {input_path}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to load memory data: {str(e)}")
            return False


# ============================================================================
# Experience Generation Module
# ============================================================================
class ExperienceGenerator:
    """
    Manager class for orchestrating experience generation from experimental results.

    This class acts as a high-level manager that:
    1. Initializes and manages the ExperienceAgent
    2. Coordinates between MemoryModule and ExperienceAgent
    3. Saves experience libraries (positive/negative)
    4. Provides a simple interface for the main pipeline

    All LLM calls and experience generation logic are delegated to the ExperienceAgent.
    """

    def __init__(self, logger: Optional[logging.Logger] = None, config_path: Optional[Dict] = None):
        """
        Initialize the experience generator manager.

        Args:
            logger: Logger instance
            config: Configuration dict containing model settings
        """
        self.logger = logger or logging.getLogger("ExperienceGenerator")
        self.config = self._load_config(config_path)
        self.experience_agent = None
        self.model_factory = ModelFactory()
        self.agent_factory = AgentFactory()


    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """
        Load configuration from file or dictionary.

        Args:
            config_path: Path to configuration file
            config: Configuration dictionary

        Returns:
            Loaded configuration dictionary
        """


        # Load from file if path provided and file exists
        if config_path and os.path.exists(config_path):
            logger.info(f"Attempting to load config from: {config_path}")
            try:
                if config_path.endswith(('.yaml', '.yml')):
                    import yaml
                    with open(config_path, 'r') as f:
                        config_data = yaml.safe_load(f)
                    logger.info(f"Successfully loaded YAML config with keys: {list(config_data.keys()) if config_data else 'None'}")
                    return config_data
                else:
                    with open(config_path, 'r') as f:
                        config_data = json.load(f)
                    logger.info(f"Successfully loaded JSON config with keys: {list(config_data.keys()) if config_data else 'None'}")
                    return config_data
            except Exception as e:
                logger.warning(f"Failed to load config from {config_path}: {str(e)}")
        elif config_path:
            logger.warning(f"Config path provided but file doesn't exist: {config_path}")
        else:
            logger.info("No config path provided")

        # Load default configuration
        default_config_path = os.path.join(os.path.dirname(__file__), "..", "..", "config", "default.yaml")
        default_config_path = os.path.abspath(default_config_path)
        logger.info(f"Loading default configuration from: {default_config_path}")

        try:
            import yaml
            with open(default_config_path, 'r') as f:
                config_data = yaml.safe_load(f)
            logger.info(f"Successfully loaded default config with keys: {list(config_data.keys()) if config_data else 'None'}")
            return config_data
        except Exception as e:
            logger.error(f"Failed to load default config from {default_config_path}: {str(e)}")
            raise RuntimeError(f"Could not load default configuration: {str(e)}")

    def _initialize_agent(self):
        """Initialize the ExperienceAgent using AgentFactory."""
        if self.experience_agent is not None:
            return

            # Create agents dict similar to interface.py
        self.experience_agent = self.agent_factory.create_agent(
            agent_type="experience",
            config=self.config,
            model_factory=self.model_factory
        )

    
        self.logger.info(f"ExperienceAgent initialized via AgentFactory")


    async def generate_experiences_from_memory(
        self,
        memory: 'MemoryModule',
        task_domain: str = "machine learning",
        output_dir: Optional[str] = None,
        learning_objective: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate experiences from all ideas and notes in memory, and update the experience library.

        This method:
        1. Collects all matched idea-experiment pairs from memory
        2. Performs contrastive learning to generate new experiences
        3. Loads existing experience library (if exists)
        4. Uses LLM to intelligently merge new experiences into the library
        5. Saves updated library and analysis results

        Args:
            memory: MemoryModule instance with loaded ideas and notes
            task_domain: Domain of the research task
            output_dir: Directory to save results (if None, defaults to results/{task_name})
            learning_objective: Optional learning objective for experience optimization

        Returns:
            Dict containing:
                - updated_library: The updated experience library
                - new_experiences: Newly generated experiences from this run
                - evaluations: Performance evaluations for each method
                - comparisons: Pairwise comparison results
                - update_stats: Statistics about library updates
        """
        self._initialize_agent()

        all_ideas = memory.get_all_ideas()
        all_experiments = memory.get_all_experiments()

        if not all_ideas:
            self.logger.warning("No ideas found in memory")
            return {"updated_library": [], "new_experiences": [], "evaluations": [], "comparisons": []}

        if not all_experiments:
            self.logger.warning("No experiment notes found in memory")
            return {"updated_library": [], "new_experiences": [], "evaluations": [], "comparisons": []}

        self.logger.info(f"Generating experiences from {len(all_ideas)} idea files and {len(all_experiments)} experiment notes")

        # Collect all matched idea-experiment pairs
        ideas_data = []
        notes_data_list = []

        for idea_entry in all_ideas:
            idea_data_list = idea_entry.get('data', [])
            if not isinstance(idea_data_list, list):
                idea_data_list = [idea_data_list]

            for idea_data in idea_data_list:
                # Try to find matching experiment notes
                idea_name = idea_data.get('name', '')

                matching_experiments = [
                    exp for exp in all_experiments
                    if exp['data'].get('name', '') == idea_name or
                       idea_name in exp.get('file_path', '')
                ]

                if not matching_experiments:
                    self.logger.warning(f"No matching experiment found for idea: {idea_name}")
                    continue

                # Use the first matching experiment
                notes_data = matching_experiments[0]['data']

                ideas_data.append(idea_data)
                notes_data_list.append(notes_data)

        if not ideas_data:
            self.logger.warning("No matched idea-experiment pairs found")
            return {"updated_library": [], "new_experiences": [], "evaluations": [], "comparisons": []}

        self.logger.info(f"Found {len(ideas_data)} matched idea-experiment pairs")

        try:
            # Step 1: Generate new experiences through contrastive learning
            self.logger.info("=" * 60)
            self.logger.info("Step 1: Generating experiences through contrastive learning")
            self.logger.info("=" * 60)

            result = await self.experience_agent.execute(
                context={
                    "ideas_data": ideas_data,
                    "notes_data_list": notes_data_list,
                    "task_domain": task_domain
                },
                params={
                    "include_comparisons": True,
                    "include_evaluations": True
                }
            )

            new_experiences = result.get("experiences", [])
            evaluations = result.get("evaluations", [])
            comparisons = result.get("comparisons", [])

            self.logger.info(f"Generated {len(new_experiences)} new experiences from {len(comparisons)} pairwise comparisons")
            # Step 2: Load existing experience library
            self.logger.info("=" * 60)
            self.logger.info("Step 2: Loading existing experience library")
            self.logger.info("=" * 60)

            existing_library = []
            library_path = None

            if output_dir:
                library_path = osp.join(output_dir, "experience_library.json")
                if osp.exists(library_path):
                    try:
                        with open(library_path, 'r', encoding='utf-8') as f:
                            library_data = json.load(f)
                            existing_library = library_data.get("experiences", [])
                        self.logger.info(f"Loaded {len(existing_library)} existing experiences from {library_path}")
                    except Exception as e:
                        self.logger.warning(f"Failed to load existing library: {e}")
                        existing_library = []
                else:
                    self.logger.info("No existing library found, creating new one")

            # Step 3: Update experience library using LLM
            self.logger.info("=" * 60)
            self.logger.info("Step 3: Updating experience library with new experiences")
            self.logger.info("=" * 60)

            update_result = await self.experience_agent.update_experience_library(
                existing_experiences=existing_library,
                new_experiences=new_experiences,
                task_domain=task_domain,
                learning_objective=learning_objective
            )

            updated_library = update_result.get("updated_library", [])
            operations = update_result.get("operations", [])
            update_metadata = update_result.get("metadata", {})

            # Step 4: Save all results
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)

                # Save updated experience library
                library_output = {
                    "experiences": updated_library,
                    "metadata": {
                        "task_domain": task_domain,
                        "total_count": len(updated_library),
                        "last_updated": datetime.now().isoformat(),
                        "learning_objective": learning_objective or f"Improve research methods in {task_domain}"
                    }
                }

                if library_path:
                    with open(library_path, 'w', encoding='utf-8') as f:
                        json.dump(library_output, f, indent=4, ensure_ascii=False)
                    self.logger.info(f"Updated experience library saved to: {library_path}")

                # Save detailed analysis results
                analysis_path = osp.join(output_dir, "experience_analysis.json")
                analysis_output = {
                    "new_experiences": new_experiences,
                    "evaluations": evaluations,
                    "comparisons": comparisons,
                    "update_operations": operations,
                    "metadata": {
                        "timestamp": datetime.now().isoformat(),
                        "task_domain": task_domain,
                        "num_ideas": len(ideas_data),
                        "num_comparisons": len(comparisons),
                        "num_new_experiences": len(new_experiences),
                        **update_metadata
                    }
                }

                with open(analysis_path, 'w', encoding='utf-8') as f:
                    json.dump(analysis_output, f, indent=4, ensure_ascii=False)
                self.logger.info(f"Detailed analysis saved to: {analysis_path}")

            # Log summary
            self.logger.info("=" * 60)
            self.logger.info("Experience Generation and Update Summary")
            self.logger.info("=" * 60)
            self.logger.info(f"Total ideas analyzed: {len(ideas_data)}")
            self.logger.info(f"Pairwise comparisons performed: {len(comparisons)}")
            self.logger.info(f"New experiences generated: {len(new_experiences)}")
            self.logger.info(f"")
            self.logger.info(f"Experience Library Update:")
            self.logger.info(f"  Original library size: {update_metadata.get('original_count', 0)}")
            self.logger.info(f"  Final library size: {update_metadata.get('final_count', 0)}")
            self.logger.info(f"  Operations performed:")
            stats = update_metadata.get('stats', {})
            self.logger.info(f"    - ADD: {stats.get('add', 0)} new experiences")
            self.logger.info(f"    - UPDATE: {stats.get('update', 0)} experiences improved")
            self.logger.info(f"    - DELETE: {stats.get('delete', 0)} experiences removed")
            self.logger.info(f"    - NONE: {stats.get('none', 0)} experiences skipped (redundant)")
            self.logger.info("=" * 60)

            # Log evaluation summary
            improved_count = sum(1 for e in evaluations if e.get("evaluation", {}).get("has_improvement", False))
            self.logger.info(f"")
            self.logger.info(f"Method Performance Summary:")
            self.logger.info(f"  Methods with improvement: {improved_count}/{len(evaluations)}")
            for eval_entry in evaluations:
                idea_name = eval_entry.get("idea_name", "unknown")
                evaluation = eval_entry.get("evaluation", {})
                if evaluation.get('has_improvement', False):
                    self.logger.info(f"  ✓ {idea_name}: Improved (Run {evaluation.get('best_run', 0)})")
                else:
                    self.logger.info(f"  ✗ {idea_name}: No improvement")
            self.logger.info("=" * 60)

            return {
                "updated_library": updated_library,
                "new_experiences": new_experiences,
                "evaluations": evaluations,
                "comparisons": comparisons,
                "update_stats": update_metadata
            }

        except Exception as e:
            self.logger.error(f"Failed to generate experiences: {str(e)}")
            import traceback
            traceback.print_exc()
            return {
                "updated_library": [],
                "new_experiences": [],
                "evaluations": [],
                "comparisons": [],
                "update_stats": {}
            }
