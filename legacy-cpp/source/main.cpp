/*
 * main.cpp
 *
 *  Created on: 21/11/2012
 *      Author: LBiC
 */

#include <iostream>
#include <fstream>
#include <cstdlib>
#include <string>
#include <ctime>

#include <boost/locale.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/date_time/gregorian/gregorian.hpp>
#include "../headers/operadores_cruzamento.h"
#include "../headers/operadores_mutacao.h"
#include "../headers/aprendizado_operadores.h"
#include "../headers/statistics.h"
#include "../headers/portfolio.h"
#include "../headers/kalman_filter.h"


using namespace boost;
//using namespace boost::locale;
using namespace boost::gregorian;

static unsigned int tam_pop = 50;
static unsigned int num_experimentos = 30;
static unsigned int num_geracoes = 50;
static std::string algo, data_base;


unsigned int exp_id;
// Prototipos de funcoes e procedimentos
void save_portfolio(const portfolio &P, std::ostream &out);
void salvar_resultados(unsigned int e, unsigned int semente_pop_inicial,
	unsigned int semente_experimento, std::vector<sol*> &P, unsigned int t);

void executa_nsga2(unsigned int e, unsigned int semente_pop_inicial,
	unsigned int semente_experimento, unsigned int tam_pop, unsigned int num_geracoes);

void executa_SMS_EMOA(unsigned int e, unsigned int semente_pop_inicial,
	unsigned int semente_experimento, unsigned int tam_pop, unsigned int num_geracoes);

void executa_algoritmo(unsigned int e, unsigned int semente_pop_inicial,
	unsigned int semente_experimento, unsigned int tam_pop, unsigned int num_geracoes);

void escrever_estatisticas(std::vector<sol*> &P, std::ostream &out, unsigned int t);
void exibe_pop(std::vector<sol*> &P, std::ostream &out);
//void ler_pop(std::vector<sol*> &P, std::ostream &in);

void load_params (int argc, char** argv)
{
	if (argc != 12)
	{
		std::cout << "Usage: " << argv[0] << " market num_assets algorithm regularization_type robustness max_cardinality start_date end_date training_percentage window_size exp_id";
		exit(0);
	}

	data_base = std::string(argv[1]);
	unsigned int num_assets = lexical_cast<unsigned int>(argv[2]);
	algo = std::string(argv[3]);
	solution::regularization_type = std::string(argv[4]);
	portfolio::robustness = lexical_cast<unsigned int>(argv[5]);
	portfolio::max_cardinality = lexical_cast<unsigned int>(argv[6]);
	portfolio::training_start_date = date(from_simple_string(argv[7]));
	portfolio::validation_end_date = date(from_simple_string(argv[8]));
	double training_percentage = lexical_cast<double>(argv[9]);
	portfolio::window_size = lexical_cast<unsigned int>(argv[10]);
	exp_id = lexical_cast<unsigned int>(argv[11]);

	if (portfolio::max_cardinality > num_assets)
	{
		std::cout << "Error: max_cardinality > num_assets\n";
		exit(0);
	}

	//start_date >> portfolio::training_start_date;
	//end_date >> portfolio::validation_end_date;
	std::cout << boost::locale::as::date << portfolio::training_start_date << ", " << portfolio::validation_end_date << std::endl;

	if (portfolio::training_start_date.day_of_week().as_enum() == Saturday)
		portfolio::training_start_date = portfolio::training_start_date + days(2);
	else if (portfolio::training_start_date.day_of_week().as_enum() == Sunday)
		portfolio::training_start_date = portfolio::training_start_date + days(1);

	if (portfolio::validation_end_date.day_of_week().as_enum() == Saturday)
			portfolio::validation_end_date = portfolio::validation_end_date - days(1);
	else if (portfolio::training_start_date.day_of_week().as_enum() == Sunday)
		portfolio::validation_end_date = portfolio::validation_end_date - days(2);

	date_period period(portfolio::training_start_date, portfolio::validation_end_date);

	portfolio::training_end_date = portfolio::training_start_date + date_duration(period.length().days()*training_percentage);
	if (portfolio::training_end_date.day_of_week().as_enum() == Saturday)
		portfolio::training_end_date = portfolio::training_end_date - days(1);
	else if (portfolio::training_end_date.day_of_week().as_enum() == Sunday)
		portfolio::training_end_date = portfolio::training_end_date - days(2);

	portfolio::validation_start_date = portfolio::training_end_date + days(1);
	if (portfolio::validation_start_date.day_of_week().as_enum() == Saturday)
			portfolio::validation_start_date = portfolio::validation_start_date + days(2);
	else if (portfolio::validation_start_date.day_of_week().as_enum() == Sunday)
		portfolio::validation_start_date = portfolio::validation_start_date + days(1);

	portfolio::available_assets_size = num_assets;

	for (unsigned int i = 0; i < portfolio::available_assets_size; ++i)
	{
		std::string caminho = "data/" + data_base + "/table (" + boost::lexical_cast<std::string>(i) + ").csv";
		std::cout << caminho << std::endl;
		asset A = load_asset_data(caminho, boost::lexical_cast<std::string>(i));
		if (A.historical_close_price.size() > 0)
			portfolio::available_assets.push_back(A);
	}

	portfolio::tr_period = portfolio::available_assets.front().historical_close_price.size();
	portfolio::vl_period = portfolio::available_assets.front().validation_close_price.size();
	portfolio::available_assets_size = portfolio::available_assets.size();

	portfolio::init_portfolio();

	op_cruzamento::prob_op_uniforme[0] = 1.0f;
	op_mutacao::prob_op_uniforme[0] = .5f;
	op_mutacao::prob_op_uniforme[1] = .5f;

	std::cout << "tr_bd: " << portfolio::training_start_date << ", tr_ed: " << portfolio::training_end_date
			  << ", vl_bd: " << portfolio::validation_start_date << ", vl_ed: " << portfolio::validation_end_date << std::endl;


	Kalman_params::F = Eigen::MatrixXd::Zero(4,4);
	Kalman_params::F << 1.0, 0.0, 1.0, 0.0,
						0.0, 1.0, 0.0, 1.0,
						0.0, 0.0, 1.0, 0.0,
						0.0, 0.0, 0.0, 1.0;
	Kalman_params::H = Eigen::MatrixXd::Zero(2,4);
	Kalman_params::H << 1.0, 0.0, 0.0, 0.0,
						0.0, 1.0, 0.0, 0.0;

	std::cout << "num_assets: " << num_assets << " period: " << period << std::endl;
	system("pause");
}

int main (int argc, char** argv)
{
	srand((unsigned)time(0));
	load_params(argc, argv);


	std::cout << "Executando os experimentos...\n";

	// Executa os experimentos.
	for (unsigned int e = exp_id; e <= (exp_id + num_experimentos) - 1; ++e)
	{
		unsigned int semente_pop_inicial = (unsigned)time(0);
		unsigned int semente_experimento = rand();

		std::cout << "Gerando pop inicial.\n";

		portfolio::compute_statistics(portfolio::current_returns_data);
		//portfolio::sample_autocorrelation(portfolio::current_returns_data, 2.0*portfolio::window_size);

		/*if (algo == "nsga2")
			executa_nsga2(e, semente_pop_inicial, semente_experimento, tam_pop, num_geracoes);
		else if (algo == "sms")
			executa_SMS_EMOA(e, semente_pop_inicial, semente_experimento, tam_pop, num_geracoes);*/

		executa_algoritmo(e, semente_pop_inicial, semente_experimento, tam_pop, num_geracoes);

	}

	return 0;
}

void exibe_pop(std::vector<sol*> &P, std::ostream &out)
{
	sort_per_objective(P,0);
	for (unsigned int p = 0; p < P.size(); ++p)
		out << P[p]->P.ROI << " "
			<< P[p]->P.risk << " "
			<< P[p]->rank_ROI << " "
			<< P[p]->rank_risk << " "
			<< P[p]->stability << " "
			<< P[p]->alpha << " "
			<< P[p]->prediction_error << std::endl;
	std::cout << std::endl;
}


void escrever_estatisticas(std::vector<sol*> &P, std::ostream &out, unsigned int t)
{

	float media_Pareto_rank = 0.0f;
	float media_cd = 0.0f;
	float media_Delta_S = 0.0f;
	float media_robust_ROI = 0.0f;
	float media_robust_Risk = 0.0f;
	float media_non_robust_ROI = 0.0f;
	float media_non_robust_Risk = 0.0f;
	float media_cardinality = 0.0f;
	float media_entropy = 0.0f;
	float media_robustness = 0.0f;
	float media_alpha = 0.0f;
	float media_prediction_error = 0.0f;


	for (unsigned int p = 0; p < P.size(); ++p)
	{
		media_Pareto_rank += P[p]->Pareto_rank;
		if (P[p]->cd < 99999999.9f)
			media_cd += P[p]->cd;

		media_Delta_S += P[p]->Delta_S;
		media_robust_ROI += P[p]->P.robust_ROI;
		media_robust_Risk += P[p]->P.robust_risk;
		media_non_robust_ROI += P[p]->P.non_robust_ROI;
		media_non_robust_Risk += P[p]->P.non_robust_risk;
		media_cardinality += P[p]->P.cardinality;
		media_entropy += portfolio::normalized_Entropy(P[p]->P);
		media_robustness += P[p]->stability;
		media_alpha += P[p]->alpha;
		//media_prediction_error += P[p]->prediction_error;
		media_prediction_error += portfolio::prediction_error(P[p]->P,t);
	}

	media_Delta_S /= P.size();
	media_Pareto_rank /= P.size();
	media_cd /= P.size();
	media_robust_ROI /= P.size();
	media_robust_Risk /= P.size();
	media_non_robust_ROI /= P.size();
	media_non_robust_Risk /= P.size();
	media_entropy /= P.size();
	media_robustness /= P.size();
	media_alpha /= P.size();
	media_prediction_error /= P.size();

	out << media_Pareto_rank << " " << media_cd	<< " " << media_Delta_S << " "
		<< media_non_robust_ROI << " " << media_non_robust_Risk << " "
		<< media_robust_ROI << " " << media_robust_Risk << " "
		<< spread(P) << " " << media_entropy << " " << media_robustness << " "
		<< media_alpha << " " << media_prediction_error << " "
		<< hypervolume(P, -1.0, 1.0) << "\n";
}

void save_portfolio(const portfolio &P, std::ostream &out)
{
	for (unsigned int i = 0; i < portfolio::available_assets_size; ++i)
		if (P.investment(i) > 0)
			out << P.available_assets[i].id << " " << P.investment(i) << "\n";
}

void salvar_resultados(unsigned int e, unsigned int semente_pop_inicial,
	unsigned int semente_experimento, std::vector<sol*> &P, unsigned int t)
{
	// Monta o nome de caminho para o arquivo que armazenara os dados da populacao final
	std::stringstream experimento_id; experimento_id << e;
	std::stringstream robustness; robustness <<  portfolio::robustness;
	std::stringstream reg_type; reg_type << solution::regularization_type;
	std::stringstream card; card << portfolio::max_cardinality;

	std::string caminho_arquivo = "experimento_" + data_base + "_card" + card.str() + "_" + algo  + "_reg" +  reg_type.str()
								+ "_rob" + robustness.str() + "_id" + experimento_id.str() + ".dat";

	// Abre o arquivo para escrita das estatisticas da populacao final
	std::ofstream arquivo(caminho_arquivo.c_str(), std::ios::out);
	arquivo << semente_pop_inicial << " " << semente_experimento << std::endl;
	escrever_estatisticas(P, arquivo, t);
	arquivo.close();

	// Abre o arquivo para escrita da populacao final
	caminho_arquivo = "populacao_" + data_base + "_card" + card.str() + "_" + algo  + "_reg" +  reg_type.str()
								+ "_rob" + robustness.str() + "_id" + experimento_id.str() + ".dat";
	arquivo.open(caminho_arquivo.c_str(), std::ios::out);
	exibe_pop(P, arquivo);
	arquivo.close();
}


void executa_SMS_EMOA(unsigned int e, unsigned int semente_pop_inicial,
	unsigned int semente_experimento, unsigned int tam_pop, unsigned int num_geracoes)
{
	srand(semente_pop_inicial);

	op_cruzamento::probabilidade = op_cruzamento::prob_op_uniforme;
	op_mutacao::probabilidade = op_mutacao::prob_op_uniforme;

	std::cout << "Criando P_trabalho...\n";

	std::vector<sol*> P_trabalho;
	for (unsigned int p = 0; p < tam_pop; ++p)
		P_trabalho.push_back(new sol());

	srand(semente_experimento);

	std::cout << "Salvando os resultados da pop inicial.\n";

	// Salva os dados para a populacao inicial em arquivo.
	salvar_resultados(e, semente_pop_inicial, semente_experimento, P_trabalho, 0);

	std::stringstream experimento_id; experimento_id << e;
	std::stringstream robustness; robustness <<  portfolio::robustness;
	std::stringstream reg_type; reg_type << solution::regularization_type;
	std::stringstream card; card << portfolio::max_cardinality;
	std::string caminho_arquivo;

	int t = 0;

	do
	{
		// Abre o arquivo para escrita da populacao final
		std::stringstream periodo; periodo << t;

		caminho_arquivo = "saida-pop_" + data_base + "_card" + card.str() + "_" + algo  + "_reg" +  reg_type.str()
								+ "_rob" + robustness.str() + "_id" + experimento_id.str() + "_t" + periodo.str() + ".dat";
		std::ofstream arquivo_geracao(caminho_arquivo.c_str(), std::ios::app);
		exibe_pop(P_trabalho, arquivo_geracao);
		arquivo_geracao.close();

		sort_per_objective(P_trabalho,0);
		for (unsigned int i = 0; i < P_trabalho.size(); ++i)
		{
			std::stringstream port; port << i;
			std::stringstream periodo; periodo << t;
			caminho_arquivo = "pop_" + data_base + "_card" + card.str() + "_" + algo  + "_reg" +  reg_type.str()
								+ "_rob" + robustness.str() + "_id" + experimento_id.str() + "_t" + periodo.str() + "_w" + port.str() + "_begin.dat";
			std::ofstream arquivo_pbest(caminho_arquivo.c_str(), std::ios::out);
			save_portfolio(P_trabalho[i]->P,arquivo_pbest);
			arquivo_pbest.close();
		}

		for (unsigned int g = 1; g <= num_geracoes; ++g)
		{

			// Abre o arquivo para escrita da populacao final
			caminho_arquivo = "saida-stats_" + data_base + "_card" + card.str() + "_" + algo  + "_reg" +  reg_type.str()
								+ "_rob" + robustness.str() + "_id" + experimento_id.str() + ".dat";
			std::ofstream arquivo_experimento(caminho_arquivo.c_str(), std::ios::app);
			escrever_estatisticas(P_trabalho, arquivo_experimento, t);
			arquivo_experimento.close();

			// Escreve as estatisticas da populacao na geracao atual na saida padrao.
			for (unsigned int s = 0; s < P_trabalho.size(); ++s)
				uma_geracao_SMS_EMOA(P_trabalho, 0.3, 2, -1.0, 10.0, t);

			// Atualiza a melhor populacao encontrada
			float current_hypervolume = hypervolume(P_trabalho, -1.0, 1.0);

			std::cout << "current_hypervolume[" << e << "][" << t << "][" << g << "]: " << current_hypervolume << std::endl;

			compute_ranks(P_trabalho);

			//std::cout << "Ranks computed...\n";

			if (g == num_geracoes)
			{
				//std::cout << "Saving pop...";
				sort_per_objective(P_trabalho,0);
				for (unsigned int i = 0; i < P_trabalho.size(); ++i)
				{
					P_trabalho[i]->anticipation = false;
					std::stringstream port; port << i;
					std::stringstream periodo; periodo << t;
					caminho_arquivo = "pop_" + data_base + "_card" + card.str() + "_" + algo  + "_reg" +  reg_type.str()
								+ "_rob" + robustness.str() + "_id" + experimento_id.str() + "t_" + periodo.str() + "w_" + port.str() + "_end.dat";
					std::ofstream arquivo_pbest(caminho_arquivo.c_str(), std::ios::out);
					save_portfolio(P_trabalho[i]->P,arquivo_pbest);
					arquivo_pbest.close();
				}
				//std::cout << "Done.\n";
			}

			//std::cout << "Saving pop statistics...\n";
			// Abre o arquivo para escrita da populacao final
			std::stringstream geracao_id; geracao_id << g;
			std::stringstream periodo; periodo << t;
			caminho_arquivo = "saida-pop_" + data_base + "_card" + card.str() + "_" + algo  + "_reg" +  reg_type.str()
								+ "_rob" + robustness.str() + "_id" + experimento_id.str() + "_t" + periodo.str() + ".dat";
			std::ofstream arquivo_geracao(caminho_arquivo.c_str(), std::ios::app);
			exibe_pop(P_trabalho, arquivo_geracao);
			arquivo_geracao.close();
			//std::cout << "Done.\n";
		}

		t++;


		int index = t*(portfolio::window_size - 1);
		std::cout << "Preparing new training data (" << index << "," << (index + portfolio::tr_returns_data.rows()) << " of ";
		std::cout << portfolio::complete_returns_data.rows() << ")... ";
		if ((index + portfolio::tr_returns_data.rows() + 2.0*portfolio::window_size) > portfolio::complete_returns_data.rows())
		{
			std::cout << "Ending experiment...\n";
			salvar_resultados(e, semente_pop_inicial, semente_experimento, P_trabalho, t-1);

			for (unsigned int p = 0; p < P_trabalho.size(); ++p)
			{
				delete P_trabalho[p];
				P_trabalho[p] = NULL;
			}
			P_trabalho.clear();
			return;
		}
		std::cout << "Re-starting...\n";


		portfolio::current_returns_data.resize(portfolio::tr_returns_data.rows(),portfolio::available_assets_size);
		for (int i = 0; i < portfolio::tr_returns_data.rows(); ++i)
			portfolio::current_returns_data.row(i) = portfolio::complete_returns_data.row(index + i);
		for (int i = 0; i < 2.0*portfolio::window_size; ++i)
			portfolio::vl_returns_data.row(i) = portfolio::complete_returns_data.row(index + portfolio::tr_returns_data.rows() + i);
		std::cout << "Done.\n";

		portfolio::compute_statistics(portfolio::current_returns_data);
		//portfolio::sample_autocorrelation(portfolio::current_returns_data, 2.0*portfolio::window_size);

		// Re-evaluate solutions in the new environment
		/*for (unsigned int p = 0; p < P_trabalho.size(); ++p)
			delete P_trabalho[p];

		for (unsigned int p = 0; p < P_trabalho.size(); ++p)
			P_trabalho[p] = new sol();*/
		std::cout << "Re-evaluating solutions in the new environment....";

		for (unsigned int p = 0; p < P_trabalho.size(); ++p)
		{
			if (portfolio::robustness)
				portfolio::compute_robust_efficiency(P_trabalho[p]->P);
			else
				portfolio::compute_efficiency(P_trabalho[p]->P);

			//std::cout << "compute_robust_efficiency(" << p << ")... ";
			//std::cout << "evaluate_robustness(" << p << ")... ";
			P_trabalho[p]->stability = portfolio::evaluate_stability(P_trabalho[p]->P);
			//std::cout << "ok.\n";
		}
		std::cout << "done.\n";

	} while (true);

}

void executa_nsga2(unsigned int e, unsigned int semente_pop_inicial,
	unsigned int semente_experimento, unsigned int tam_pop, unsigned int num_geracoes)
{
	srand(semente_pop_inicial);

	op_cruzamento::probabilidade = op_cruzamento::prob_op_uniforme;
	op_mutacao::probabilidade = op_mutacao::prob_op_uniforme;

	std::cout << "Criando P_trabalho...\n";

	portfolio::compute_statistics(portfolio::current_returns_data);
	//portfolio::sample_autocorrelation(portfolio::current_returns_data, 2.0*portfolio::window_size);

	std::vector<sol*> P_trabalho;
	for (unsigned int p = 0; p < tam_pop; ++p)
		P_trabalho.push_back(new sol());

	// Executa o NSGA-II
	srand(semente_experimento);

	std::cout << "Salvando os resultados da pop inicial.\n";
	// Salva os dados para a populacao inicial em arquivo.
	salvar_resultados(e, semente_pop_inicial, semente_experimento, P_trabalho, 0);

	std::stringstream experimento_id; experimento_id << e;
	std::stringstream robustness; robustness <<  portfolio::robustness;
	std::stringstream reg_type; reg_type << solution::regularization_type;
	std::stringstream card; card << portfolio::max_cardinality;
	std::string caminho_arquivo;

	int t = 0;

	do
	{
		// Abre o arquivo para escrita da populacao final
		std::stringstream geracao_id; geracao_id << 0;
		std::stringstream periodo; periodo << t;
		caminho_arquivo = "saida-pop_" + data_base + "_card" + card.str() + "_" + algo  + "_reg" +  reg_type.str()
								+ "_rob" + robustness.str() + "_id" + experimento_id.str() + "_t" + periodo.str() + ".dat";
		std::ofstream arquivo_geracao(caminho_arquivo.c_str(), std::ios::app);
		exibe_pop(P_trabalho, arquivo_geracao);
		arquivo_geracao.close();

		sort_per_objective(P_trabalho,0);
		for (unsigned int i = 0; i < P_trabalho.size(); ++i)
		{
			std::stringstream port; port << i;
			std::stringstream periodo; periodo << t;
			caminho_arquivo = "pop_" + data_base + "_card" + card.str() + "_" + algo  + "_reg" +  reg_type.str()
								+ "_rob" + robustness.str() + "_id" + experimento_id.str() + "_t" + periodo.str() + "_w" + port.str() + "_begin.dat";
			std::ofstream arquivo_pbest(caminho_arquivo.c_str(), std::ios::out);
			save_portfolio(P_trabalho[i]->P,arquivo_pbest);
			arquivo_pbest.close();
		}

		for (unsigned int g = 1; g <= num_geracoes; ++g)
		{

			// Abre o arquivo para escrita da populacao final
			caminho_arquivo = "saida-stats_" + data_base + "_card" + card.str() + "_" + algo  + "_reg" +  reg_type.str()
								+ "_rob" + robustness.str() + "_id" + experimento_id.str() + ".dat";
			std::ofstream arquivo_experimento(caminho_arquivo.c_str(), std::ios::app);
			escrever_estatisticas(P_trabalho, arquivo_experimento, t);
			arquivo_experimento.close();

			// Escreve as estatisticas da populacao na geracao atual na saida padrao.
			uma_geracao_NSGAII(P_trabalho, 0.3, 2, t);

			// Atualiza a melhor populacao encontrada
			float current_hypervolume = hypervolume(P_trabalho, -1.0, 1.0);

			std::cout << "current_hypervolume[" << e << "][" << t << "][" << g << "]: " << current_hypervolume << std::endl;

			compute_ranks(P_trabalho);

			//std::cout << "Ranks computed...\n";

			if (g == num_geracoes)
			{
				//std::cout << "Saving pop...";
				sort_per_objective(P_trabalho,0);
				for (unsigned int i = 0; i < P_trabalho.size(); ++i)
				{
					std::stringstream port; port << i;
					std::stringstream periodo; periodo << t;
					caminho_arquivo = "pop_" + data_base + "_card" + card.str() + "_" + algo  + "_reg" +  reg_type.str()
						+ "_rob" + robustness.str() + "_id" + experimento_id.str() + "_t" + periodo.str() + "_w" + port.str() + "_end.dat";
					std::ofstream arquivo_pbest(caminho_arquivo.c_str(), std::ios::out);
					save_portfolio(P_trabalho[i]->P,arquivo_pbest);
					arquivo_pbest.close();
				}
				//std::cout << "Done.\n";
			}

			//std::cout << "Saving pop statistics...\n";
			// Abre o arquivo para escrita da populacao final
			std::stringstream geracao_id; geracao_id << g;
			std::stringstream periodo; periodo << t;
			caminho_arquivo = "saida-pop_" + data_base + "_card" + card.str() + "_" + algo  + "_reg" +  reg_type.str()
								+ "_rob" + robustness.str() + "_id" + experimento_id.str() + "_t" + periodo.str() + ".dat";
			std::ofstream arquivo_geracao(caminho_arquivo.c_str(), std::ios::app);
			exibe_pop(P_trabalho, arquivo_geracao);
			arquivo_geracao.close();
			//std::cout << "Done.\n";
		}


		t++;

		int index = t*(portfolio::window_size - 1);
		std::cout << "Preparing next training period (" << index << "," << (index + portfolio::tr_returns_data.rows()) << " of ";
		std::cout << portfolio::complete_returns_data.rows() << ")... ";
		if ((index + portfolio::tr_returns_data.rows()) > portfolio::complete_returns_data.rows())
		{
			std::cout << "Ending experiment...\n";
			salvar_resultados(e, semente_pop_inicial, semente_experimento, P_trabalho, t-1);

			for (unsigned int p = 0; p < P_trabalho.size(); ++p)
			{
				delete P_trabalho[p];
				P_trabalho[p] = NULL;
			}
			P_trabalho.clear();
			return;
		}
		std::cout << "Re-starting...\n";


		portfolio::current_returns_data.resize(portfolio::tr_returns_data.rows(),portfolio::available_assets_size);
		for (int i = 0; i < portfolio::tr_returns_data.rows(); ++i)
			portfolio::current_returns_data.row(i) = portfolio::complete_returns_data.row(index + i);
		std::cout << "Done.\n";

		portfolio::compute_statistics(portfolio::current_returns_data);
		//portfolio::sample_autocorrelation(portfolio::current_returns_data, 2.0*portfolio::window_size);

		// Re-evaluate solutions in the new environment
		/*for (unsigned int p = 0; p < P_trabalho.size(); ++p)
			delete P_trabalho[p];

		for (unsigned int p = 0; p < P_trabalho.size(); ++p)
			P_trabalho[p] = new sol();*/
		std::cout << "Re-evaluating solutions in the new environment....";

		for (unsigned int p = 0; p < P_trabalho.size(); ++p)
		{

			if (portfolio::robustness)
				portfolio::compute_robust_efficiency(P_trabalho[p]->P);
			else
				portfolio::compute_efficiency(P_trabalho[p]->P);

			std::cout << "ok.\n";
			std::cout << "evaluate_robustness(" << p << ")... ";
			P_trabalho[p]->stability = portfolio::evaluate_stability(P_trabalho[p]->P);
			std::cout << "ok.\n";
		}
		std::cout << "done.\n";

	} while (true);

}

void executa_algoritmo(unsigned int e, unsigned int semente_pop_inicial,
	unsigned int semente_experimento, unsigned int tam_pop, unsigned int num_geracoes)
{
	srand(semente_pop_inicial);

	op_cruzamento::probabilidade = op_cruzamento::prob_op_uniforme;
	op_mutacao::probabilidade = op_mutacao::prob_op_uniforme;

	std::cout << "Criando P_trabalho...\n";

	std::vector<sol*> P_trabalho;
	for (unsigned int p = 0; p < tam_pop; ++p)
		P_trabalho.push_back(new sol());

	srand(semente_experimento);

	std::cout << "Salvando os resultados da pop inicial.\n";

	// Salva os dados para a populacao inicial em arquivo.
	salvar_resultados(e, semente_pop_inicial, semente_experimento, P_trabalho, 0);

	std::stringstream experimento_id; experimento_id << e;
	std::stringstream robustness; robustness <<  portfolio::robustness;
	std::stringstream reg_type; reg_type << solution::regularization_type;
	std::stringstream card; card << portfolio::max_cardinality;
	std::string caminho_arquivo;

	int t = 0;

	do
	{
		// Abre o arquivo para escrita da populacao final
		std::stringstream periodo; periodo << t;

		caminho_arquivo = "saida-pop_" + data_base + "_card" + card.str() + "_" + algo  + "_reg" +  reg_type.str()
								+ "_rob" + robustness.str() + "_id" + experimento_id.str() + "_t" + periodo.str() + ".dat";
		std::ofstream arquivo_geracao(caminho_arquivo.c_str(), std::ios::app);
		exibe_pop(P_trabalho, arquivo_geracao);
		arquivo_geracao.close();

		sort_per_objective(P_trabalho,0);
		for (unsigned int i = 0; i < P_trabalho.size(); ++i)
		{
			std::stringstream port; port << i;
			std::stringstream periodo; periodo << t;
			caminho_arquivo = "pop_" + data_base + "_card" + card.str() + "_" + algo  + "_reg" +  reg_type.str()
								+ "_rob" + robustness.str() + "_id" + experimento_id.str() + "_t" + periodo.str() + "_w" + port.str() + "_begin.dat";
			std::ofstream arquivo_pbest(caminho_arquivo.c_str(), std::ios::out);
			save_portfolio(P_trabalho[i]->P,arquivo_pbest);
			arquivo_pbest.close();
		}

		for (unsigned int g = 1; g <= num_geracoes; ++g)
		{

			// Abre o arquivo para escrita da populacao final
			caminho_arquivo = "saida-stats_" + data_base + "_card" + card.str() + "_" + algo  + "_reg" +  reg_type.str()
								+ "_rob" + robustness.str() + "_id" + experimento_id.str() + ".dat";
			std::ofstream arquivo_experimento(caminho_arquivo.c_str(), std::ios::app);
			escrever_estatisticas(P_trabalho, arquivo_experimento, t);
			arquivo_experimento.close();

			if (algo == "sms")
			{
				// Escreve as estatisticas da populacao na geracao atual na saida padrao.
				for (unsigned int s = 0; s < P_trabalho.size(); ++s)
					uma_geracao_SMS_EMOA(P_trabalho, 0.3, 2, -1.0, 10.0, t);
			}
			else if (algo == "nsga2")
			{
				// Escreve as estatisticas da populacao na geracao atual na saida padrao.
				uma_geracao_NSGAII(P_trabalho, 0.3, 2, t);
			}

			// Atualiza a melhor populacao encontrada
			float current_hypervolume = hypervolume(P_trabalho, -1.0, 1.0);

			std::cout << "current_hypervolume[" << e << "][" << t << "][" << g << "]: " << current_hypervolume << std::endl;

			compute_ranks(P_trabalho);

			//std::cout << "Ranks computed...\n";

			if (g == num_geracoes)
			{
				//std::cout << "Saving pop...";
				sort_per_objective(P_trabalho,0);
				for (unsigned int i = 0; i < P_trabalho.size(); ++i)
				{
					P_trabalho[i]->anticipation = false;
					std::stringstream port; port << i;
					std::stringstream periodo; periodo << t;
					caminho_arquivo = "pop_" + data_base + "_card" + card.str() + "_" + algo  + "_reg" +  reg_type.str()
								+ "_rob" + robustness.str() + "_id" + experimento_id.str() + "t_" + periodo.str() + "w_" + port.str() + "_end.dat";
					std::ofstream arquivo_pbest(caminho_arquivo.c_str(), std::ios::out);
					save_portfolio(P_trabalho[i]->P,arquivo_pbest);
					arquivo_pbest.close();
				}
				//std::cout << "Done.\n";
			}

			//std::cout << "Saving pop statistics...\n";
			// Abre o arquivo para escrita da populacao final
			std::stringstream geracao_id; geracao_id << g;
			std::stringstream periodo; periodo << t;
			caminho_arquivo = "saida-pop_" + data_base + "_card" + card.str() + "_" + algo  + "_reg" +  reg_type.str()
								+ "_rob" + robustness.str() + "_id" + experimento_id.str() + "_t" + periodo.str() + ".dat";
			std::ofstream arquivo_geracao(caminho_arquivo.c_str(), std::ios::app);
			exibe_pop(P_trabalho, arquivo_geracao);
			arquivo_geracao.close();
			//std::cout << "Done.\n";
		}

		t++;


		int index = t*(portfolio::window_size - 1);
		std::cout << "Preparing new training data (" << index << "," << (index + portfolio::tr_returns_data.rows()) << " of ";
		std::cout << portfolio::complete_returns_data.rows() << ")... ";
		if ((index + portfolio::tr_returns_data.rows() + 2.0*portfolio::window_size) > portfolio::complete_returns_data.rows())
		{
			std::cout << "Ending experiment...\n";
			salvar_resultados(e, semente_pop_inicial, semente_experimento, P_trabalho, t-1);

			for (unsigned int p = 0; p < P_trabalho.size(); ++p)
			{
				delete P_trabalho[p];
				P_trabalho[p] = NULL;
			}
			P_trabalho.clear();
			return;
		}
		std::cout << "Re-starting...\n";


		portfolio::current_returns_data.resize(portfolio::tr_returns_data.rows(),portfolio::available_assets_size);
		for (int i = 0; i < portfolio::tr_returns_data.rows(); ++i)
			portfolio::current_returns_data.row(i) = portfolio::complete_returns_data.row(index + i);
		for (int i = 0; i < 2.0*portfolio::window_size; ++i)
			portfolio::vl_returns_data.row(i) = portfolio::complete_returns_data.row(index + portfolio::tr_returns_data.rows() + i);
		std::cout << "Done.\n";

		portfolio::compute_statistics(portfolio::current_returns_data);
		//portfolio::sample_autocorrelation(portfolio::current_returns_data, 2.0*portfolio::window_size);

		// Re-evaluate solutions in the new environment
		/*for (unsigned int p = 0; p < P_trabalho.size(); ++p)
			delete P_trabalho[p];

		for (unsigned int p = 0; p < P_trabalho.size(); ++p)
			P_trabalho[p] = new sol();*/
		std::cout << "Re-evaluating solutions in the new environment....";

		for (unsigned int p = 0; p < P_trabalho.size(); ++p)
		{
			if (portfolio::robustness)
				portfolio::compute_robust_efficiency(P_trabalho[p]->P);
			else
				portfolio::compute_efficiency(P_trabalho[p]->P);

			//std::cout << "compute_robust_efficiency(" << p << ")... ";
			//std::cout << "evaluate_robustness(" << p << ")... ";
			P_trabalho[p]->stability = portfolio::evaluate_stability(P_trabalho[p]->P);
			//std::cout << "ok.\n";
		}
		std::cout << "done.\n";

	} while (true);

}


