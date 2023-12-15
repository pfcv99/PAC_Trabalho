# %% [markdown]
# # Projeto de Programação e Algoritmos em Ciência

# %% [markdown]
# ## Tendências globais em matéria de saúde mental.

# %% [markdown]
# ### 1. Introdução 

# %% [markdown]
# Este conjunto de dados contém informações de países do mundo sobre a prevalência de distúrbios de saúde mental, incluindo a esquizofrenia, o transtorno bipolar, distúrbios alimentares, transtornos de ansiedade, transtornos por uso de drogas, depressão e transtornos por uso de álcool. Estes dados poderão ser úteis na medida em que poderá ser possível obter insights sobre como essas questões estão a impactar as vidas das pessoas. Há, portanto, algumas questões que se levantam, tais como:
# 
# 1. Quais são os tipos de distúrbios de saúde mental que as pessoas pelo mundo têm vindo a enfrentar?
# 2. Quantas pessoas em cada país sofrem de problemas de saúde mental?
# 3. Serão os homens ou mulheres quem têm maior probabilidade de ter depressão?
# 4. A depressão está relacionada com o suicídio?
# 
# Ao explorar padrões entre as taxas de prevalência por meio de visualização de dados, é, portanto, possível compreender melhor essas questões complexas.

# %% [markdown]
# ### 2. Métodos 

# %% [markdown]
# Neste trabalho optou-se por se subdividir o problema em quatro partes:
# 
# 1. Obtenção dos dados
# 2. Tratamento dos dados
# 3. Análise Exploratória 
# 4. Configuração do menu e função principal
# 
# Cada parte encontra-se documentada com comentários nos quais são detalhados todos os passos efetuados. 

# %% [markdown]
# #### 2.1. Obtenção dos dados

# %% [markdown]
# O [dataset](https://www.kaggle.com/datasets/thedevastator/uncover-global-trends-in-mental-health-disorder) usado foi obtido através do website [Kaggle](https://www.kaggle.com/), um dos maiores repositórios de modelos, dados e código publicados pela comunidade.

# %% [markdown]
# #### 2.2. Tratamento dos dados

# %% [markdown]
# Nesta parte fez-se um conjunto de operações para tornar o dataset 'usable'. Se virmos este dataset como uma tabela, então a tabela principal era composta por sub-tabelas que se encontravam acopladas e sem uma estrutura coesa e organizada. Foi, portanto, necessário efetuar operações que limpeza, reorganização e criação de novos dataframes.

# %% [markdown]
# ##### Módulos Usados

# %%
import os
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.subplots as sp
import plotly.graph_objects as go
from ipywidgets import widgets
from art import *
from tabulate import tabulate
from functools import partial

# %% [markdown]
# ##### 2.2.1. Upload dos dados em bruto

# %%
#Carregar o conjunto de dados principal a partir do arquivo CSV especificado ('Mental_health_Depression_disorder_Data.csv')
dataset = pd.read_csv('data/raw_data/Mental_health_Depression_disorder_Data.csv')

# Carregar outro conjunto de dados contendo códigos de país ISO a partir do arquivo CSV especificado ('iso_countries.csv')
iso_countries = pd.read_csv('data/raw_data/iso_countries.csv')

# %% [markdown]
# ##### 2.2.2. Criação dos dataframes

# %%
# Criar um novo DataFrame do pandas chamado 'df_geral' e preenchê-lo com os dados do conjunto de dados principal
df_geral = pd.DataFrame(dataset)

# Criar um novo DataFrame do pandas chamado 'df_iso' e preenchê-lo com os dados do conjunto de dados de códigos de país ISO
df_iso = pd.DataFrame(iso_countries)

# %% [markdown]
# ##### 2.2.3. Identificação das sub-tabelas

# %%
# Criar uma visualização de subconjunto (subtable_view) do DataFrame df_geral, selecionando apenas as linhas em que a coluna 'Year' é igual à string 'Year' para identificar os cabeçalhos de coluna
subtable_view = df_geral[df_geral['Year'] == 'Year']

# Exibir as primeiras linhas da visualização de subconjunto usando o método head(), que por padrão mostra as primeiras 5 linhas
subtable_view.head()


# %% [markdown]
# ##### 2.2.4. Identificação dos índices para separação das tabelas

# %%
# Criar uma lista (index_lst) contendo os valores da coluna 'index' da visualização de subconjunto (subtable_view)
index_lst = subtable_view['index'].tolist()

# Inserir o valor 0 no início da lista
index_lst.insert(0, 0)

# Adicionar o último valor da coluna 'index' do DataFrame original (df_geral) ao final da lista
index_lst.append(df_geral['index'].iloc[-1])

# Imprimir a lista resultante
print(index_lst)


# %% [markdown]
# ##### 2.2.5. Operações de limpeza do dataframe principal, criação de novos dataframes a partir das sub-tabelas e armazenamento dos dataframes limpos em ficheiros CSV

# %%
# Criar um dicionário vazio para armazenar os DataFrames resultantes
dfs = {}

# Iterar sobre os elementos da lista de índices
for i in range(len(index_lst) - 1):
    # Definir o início e o final do intervalo com base nos valores da lista de índices
    start = index_lst[i]
    end = index_lst[i + 1] - 1  # Subtrai 1 para obter o intervalo correto

    # Criar um novo DataFrame (dfs[f'df_{i}']) com base no intervalo definido
    dfs[f'df_{i}'] = df_geral.iloc[start:end + 1].copy()

    # Fazer reset dos índices e remover as colunas que são compostas apenas por valores nulos (NaN)
    dfs[f'df_{i}'].reset_index(drop=True, inplace=True)
    dfs[f'df_{i}'] = dfs[f'df_{i}'].dropna(axis=1, how='all')
    
    # Substituir 'NaN' na coluna 'Code' pelo valor correspondente na coluna 'Entity'
    dfs[f'df_{i}']['Code'].fillna(dfs[f'df_{i}']['Entity'], inplace=True)

    # Remover linhas com valores nulos nas colunas
    dfs[f'df_{i}'] = dfs[f'df_{i}'].dropna(subset=dfs[f'df_{i}'].columns)

    # Renomear as colunas com base na primeira linha, exceto para o primeiro DataFrame
    if i > 0:
        col_names = dfs[f'df_{i}'].iloc[0].tolist()
        dfs[f'df_{i}'].columns = col_names

        # Remover a linha que foi usada como nome de colunas
        dfs[f'df_{i}'] = dfs[f'df_{i}'].iloc[1:].reset_index(drop=True)
    
    # Adicionar uma nova coluna 'region' ao DataFrame com base na coluna 'Code' e no DataFrame df_iso
    dfs[f'df_{i}']['Continent'] = dfs[f'df_{i}']['Code'].map(df_iso.set_index('alpha-3')['region'])
    
    # Filtrar linhas onde 'Code' não está presente em df_iso['alpha-3']
    dfs[f'sub_df_{i}'] = dfs[f'df_{i}'][~dfs[f'df_{i}']['Code'].isin(df_iso['alpha-3'].str.upper())].reset_index(drop=True)
    
    # Filtrar linhas onde 'Code' está presente em df_iso['alpha-3'] e guardar num novo DataFrame
    dfs[f'df_{i}'] = dfs[f'df_{i}'][dfs[f'df_{i}']['Code'].isin(df_iso['alpha-3'])].reset_index(drop=True)
    
    # Guardar o DataFrame como ficheiro CSV com um nome correspondente ao índice
    dfs[f'df_{i}'].to_csv(f'data/clean_data/df_{i}.csv', index=False)
    dfs[f'sub_df_{i}'].to_csv(f'data/clean_data/sub_df_{i}.csv', index=False)


# %% [markdown]
# #### 2.3. Análise exploratória

# %% [markdown]
# Nesta secção, optou-se por criar um conjunto de gráficos e tabelas que permitissem uma melhor compreensão do dataset, bem como encontrar relações entre as diversas variáveis.

# %% [markdown]
# ##### Dataframes limpos

# %%
# Carregar o ficheiro CSV 'df_0.csv' localizado em 'data/clean_data/' para um DataFrame chamado df_0
data_df_0 = pd.read_csv('data/clean_data/df_0.csv')
df_0 = pd.DataFrame(data_df_0)

# Carregar o ficheiro CSV 'df_1.csv' localizado em 'data/clean_data/' para um DataFrame chamado df_1
data_df_1 = pd.read_csv('data/clean_data/df_1.csv')
df_1 = pd.DataFrame(data_df_1)

# Carregar o ficheiro CSV 'sub_df_1.csv' localizado em 'data/clean_data/' para um DataFrame chamado sub_df_1
data_sub_df_1 = pd.read_csv('data/clean_data/sub_df_1.csv')
sub_df_1 = pd.DataFrame(data_sub_df_1)

# Carregar o ficheiro CSV 'df_2.csv' localizado em 'data/clean_data/' para um DataFrame chamado df_2
data_df_2 = pd.read_csv('data/clean_data/df_2.csv')
df_2 = pd.DataFrame(data_df_2)

# Carregar o ficheiro CSV 'df_3.csv' localizado em 'data/clean_data/' para um DataFrame chamado df_3
data_df_3 = pd.read_csv('data/clean_data/df_3.csv')
df_3 = pd.DataFrame(data_df_3)

# %% [markdown]
# ##### 2.3.1. Gráficos

# %% [markdown]
# A. Mapa do mundo com estatísticas para a depressão ao longo do tempo

# %%
import plotly.express as px

def mapa():
    # Criar um gráfico de mapa de choropleth utilizando o Plotly Express
    fig = px.choropleth(df_0,  # DataFrame de dados
                        locations='Code',  # Coluna do DataFrame que contém os códigos de localização
                        color='Depression (%)',  # Coluna do DataFrame que contém os valores para atribuir cor
                        scope="world",  # Scope geográfico do mapa (neste caso, mundial)
                        hover_name='Entity',  # Coluna do DataFrame usada para rótulos ao passar o rato sobre as áreas
                        color_continuous_scale=px.colors.sequential.Plasma,  # Esquema de cores contínuas
                        animation_frame='Year',  # Coluna do DataFrame usada para a animação ao longo do tempo
                        animation_group='Entity')  # Coluna do DataFrame usada para agrupar áreas durante a animação

    # Atualizar o layout do gráfico
    fig.update_layout(
        title=f"Mapa Mundial da Depressão (%) ao longo dos anos",
        font=dict(family="Arial", size=12),
        margin=dict(l=5, r=5, t=50, b=5),
        coloraxis=dict(cmin=df_0['Depression (%)'].min(), cmax=df_0['Depression (%)'].max()),  # Configurar os limites da escala de cores
        height=700
    )
    
    # Atualizar as configurações geográficas para ajustar os limites de exibição
    fig.update_geos(fitbounds="locations", visible=False)
    
    # Exibir o gráfico
    fig.show()


# %% [markdown]
# B.  Gráfico da relação da Taxa de Depressão com a Taxa de Suicídio no mundo

# %%
def bolhas():
    # Criar um gráfico de dispersão com bolhas utilizando o Plotly Express
    fig = px.scatter(
        df_2,  # DataFrame de dados
        x="Depressive disorder rates (number suffering per 100,000)",  # Eixo x: Taxa de transtorno depressivo
        y="Suicide rate (deaths per 100,000 individuals)",  # Eixo y: Taxa de suicídio
        animation_frame="Year",  # Coluna do DataFrame usada para a animação ao longo do tempo
        animation_group="Entity",  # Coluna do DataFrame usada para agrupar pontos durante a animação
        size="Population",  # Tamanho das bolhas baseado na coluna 'Population'
        color="Continent",  # Cor das bolhas baseada na coluna 'Continent'
        hover_name="Entity",  # Rótulo ao passar o rato sobre as bolhas
        facet_col="Continent",  # Criar subgráficos separados por continente
        log_x=True,  # Usar escala logarítmica no eixo x
        size_max=50,  # Tamanho máximo das bolhas
        range_x=[2000, 6000],  # Faixa de valores no eixo x
        range_y=[0, 60]  # Faixa de valores no eixo y
    )

    # Atualizar os eixos x
    fig.update_xaxes(
        tickangle=90,  # Ângulo de inclinação dos rótulos no eixo x
        title_text="Taxa de Transtorno Depressivo",  # Título do eixo x
        title_font={"size": 12},  # Tamanho da fonte do título do eixo x
        title_standoff=25  # Distância entre o título e o eixo x
    )

    # Atualizar o layout do gráfico
    fig.update_layout(
        title=dict(text="Gráfico ilustrando a relação global entre Taxas de Depressão e Suicídio", font=dict(size=20), yref='paper')  # Título do gráfico
    )

    # Exibir o gráfico
    fig.show()

# %% [markdown]
# C. Gráfico de barras com a Pervalência de Depressão em homens e mulheres no mundo (Top 20)

# %%
def barras():
    # Criar uma figura e eixo com um tamanho específico
    fig, ax = plt.subplots(figsize=(6, 6))
    
    # Entrada para o ano
    year = int(input('Ano: '))
    
    # Filtrar dados para o ano especificado
    data_year = df_1[df_1['Year'] == year]
    
    # Calcular a prevalência média para homens e mulheres
    data_year['Média da Prevalência'] = (data_year['Prevalence in males (%)'] + data_year['Prevalence in females (%)']) / 2
    
    # Ordenar por prevalência média em ordem decrescente
    data_year = data_year.sort_values(by='Média da Prevalência', ascending=False).head(20)
    
    # Criar gráficos de barras para homens e mulheres separadamente
    bar_height = 0.8 
    ax.barh(data_year['Entity'], data_year['Prevalence in males (%)'], height=bar_height, color='blue', alpha=0.3, label='Homens')
    ax.barh(data_year['Entity'], data_year['Prevalence in females (%)'], left=data_year['Prevalence in males (%)'], height=bar_height, color='darkorange', alpha=0.3, label='Mulheres')
    
    # Adicionar rótulos de percentagem para homens
    for i, valor in enumerate(data_year['Prevalence in males (%)']):
        ax.text(valor, i, f'{valor:.2f}%', ha='right', va='center', color='blue', fontweight='bold')
    
    # Adicionar rótulos de percentagem para mulheres
    for i, valor in enumerate(data_year['Prevalence in females (%)']):
        ax.text(valor, i, f'{valor:.2f}%', ha='left', va='center', color='darkorange', fontweight='bold')
    
    # Adicionar uma barra para o mundo
    world_values = sub_df_1[(sub_df_1['Entity'] == 'World') & (sub_df_1['Year'] == year)]
    world_male_value = world_values['Prevalence in males (%)'].values[0]
    world_female_value = world_values['Prevalence in females (%)'].values[0]
    ax.barh('Mundo',world_male_value, height=bar_height, color='#00E676', alpha=0.3, label='Mundo Homens')
    ax.barh('Mundo', world_female_value, left=world_male_value, height=bar_height, color='#CDDC39', alpha=0.3, label='Mundo Mulheres')

    # Adicionar rótulos de percentagem para o mundo
    ax.text(world_male_value, len(data_year), f'{world_male_value:.2f}%', ha='right', va='center', color='#00E676', fontweight='bold')
    ax.text(world_female_value, len(data_year), f'{world_female_value:.2f}%', ha='left', va='center', color='#CDDC39', fontweight='bold')

    # Adicionar um título e rótulos aos eixos
    ax.set_title(f'Prevalência de depressão em homens e mulheres para o ano {year}', weight='bold')
    ax.set_xlabel('Prevalência (%)')
    ax.set_ylabel('País')
    
    # Inverter o eixo y para ter barras maiores no topo
    ax.invert_yaxis()
    
    # Exibir a legenda
    ax.legend()
    
    # Exibir o gráfico
    plt.show()
    
    # Limpar a tela
    clear_screen()
    
    # Retornar True para indicar que a tabela foi chamada
    return True



# %% [markdown]
# D. Evolução das doenças mentais ao longo dos anos

# %%
def linhas():
    # Solicitar o país
    country = input('Qual o país: ').capitalize()
    selected_country = df_0[df_0['Entity'] == country]

    # Ajustar o tamanho dos subplots
    fig = sp.make_subplots(rows=selected_country.columns[5:11].shape[0], cols=1, subplot_titles=selected_country.columns[5:11],
                           shared_xaxes=True, vertical_spacing=0.02)

    # Adicionar linhas para cada subplot
    for i, col in enumerate(selected_country.columns[5:11], start=1):
        fig.add_trace(go.Scatter(x=selected_country["Year"], y=selected_country[col], mode='lines', name=col),
                      row=i, col=1)

    # Atualizar layout
    fig.update_layout(height=selected_country.columns[5:11].shape[0] * 200, title_text='Evolution of the main mental health issues over the years',
                      showlegend=False)

    # Definir dtick para cada gráfico
    dtick_values = [0.001, 0.02, 0.01, 0.01, 0.1, 0.02]  # Precisa de ser otimizado

    for i in range(1, selected_country.columns[5:11].shape[0] + 1):
        fig.update_yaxes(tickmode='linear', dtick=dtick_values[i-1], row=i, col=1, showgrid=False)

    # Atualizar eixos
    fig.update_xaxes(dtick="M1", tickformat="%Y", ticklabelmode="period")

    # Mostrar o gráfico
    fig.show()


# %% [markdown]
# ##### 2.3.2. Tabelas

# %% [markdown]
# A. Função para obter as tabelas dos dataframes

# %%
def tabela(df):
    # Solicitar o país
    country = input('Qual o país: ').capitalize()
    
    # Imprimir o nome do país
    print("=" * len(country), country, "=" * len(country), sep="\n")
    
    # Imprimir a tabela usando o tabulate
    print(tabulate(df[df['Entity'] == country], headers='keys', tablefmt='fancy_grid'))
    
    # Aguardar que o utilizador pressione de Enter para continuar
    input("Pressione Enter para continuar...")
    
    # Limpar a tela após pressionar Enter
    clear_screen()
    
    # Retornar True para indicar que a tabela foi chamada
    return True

# %% [markdown]
# B. Função para obter as tabelas estatísticas

# %%
def tabela_describe(df):
    # Solicitar o país
    country = input('Qual o país: ').capitalize()
    
    # Imprimir o nome do país
    print("=" * len(country), country, "=" * len(country), sep="\n")
    
    # Imprimir a tabela usando o tabulate
    print(tabulate(df[df['Entity'] == country].describe(), headers='keys', tablefmt='fancy_grid'))
    
    # Aguardar que o utilizador pressione de Enter para continuar
    input("Pressione Enter para continuar...")
    
    # Limpar a tela após pressionar Enter
    os.system('cls' if os.name == 'nt' else 'clear')
    
    # Retornar True para indicar que a tabela foi chamada
    return True

# %% [markdown]
# #### 2.4. Configuração do menu e função principal

# %% [markdown]
# ##### 2.4.1. Menu

# %%
def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

# %%
def menu(titulo, opcoes):
    clear_flag = True  # Flag inicial para limpar a tela

    while True: # Inicia um loop infinito para manter o menu em execução até que o utilizador escolha a opção de sair.
        if clear_flag: # Verifica se a flag para limpar a tela está ativa.
            clear_screen()

        # Apresentação do cabeçalho do projeto
        print("=" * len('PROJETO DE PROGRAMAÇÃO E ALGORITMOS EM CIÊNCIAS'), 'PROJETO DE PROGRAMAÇÃO E ALGORITMOS EM CIÊNCIAS\n\nPedro Venâncio, nº88226\nMestrado em Bioinformática Clínica\n', sep="\n")
        print("=" * len(titulo), titulo, "=" * len(titulo), sep="\n")

        # Apresentação das opções do menu
        for i, (opcao, funcao) in enumerate(opcoes, 1): # Itera sobre as opções do menu com um índice começado em 1
            print("[{}] - {}".format(i, opcao)) # Apresenta cada opção numerada
        print("[{}] - Voltar/Sair".format(i + 1)) # Apresenta a opção para voltar ou sair do menu.
        op = input("Opção: ") # Solicita a entrada do utilizador

        # Verificação se a entrada é um número válido
        if op.isdigit():
            if 1 <= int(op) <= i + 1:
                if int(op) == i + 1:
                    # Encerra este menu e retorna à função anterior
                    break
                if int(op) <= len(opcoes):
                    # Chama a função associada à opção escolhida
                    clear_flag = opcoes[int(op) - 1][1]()
                    continue

        # Mensagem de erro para entrada inválida
        print("Opção inválida. \n\n")
        clear_flag = True

    return clear_flag


# %% [markdown]
# ##### 2.4.2. Função para aceder aos gráficos no menu

# %%
def graficos():
    # Lista de opções para o menu de gráficos, cada opção é um tuplo contendo o nome do gráfico e a função associada
    opcoes = [
        ("Mapa do mundo com estatísticas para a depressão ao longo do tempo", mapa),
        ("Gráfico da relação da Taxa de Depressão com a Taxa de Suicídio no mundo", bolhas),
        ("Gráfico de barras com a prevalência de depressão em homens e mulheres", barras),
        ("Gráfico de linhas com a evolução dos principais problemas de saúde mental ao longo dos anos", linhas)
    ]

    # Chama a função do menu e passa o título 'Gráficos' e a lista de opções
    return menu('Gráficos', opcoes)


# %% [markdown]
# ##### 2.4.3. Função para aceder às tabelas no menu

# %%
def tabelas():
    # Lista de opções para o menu de tabelas, cada opção é um tuplo contendo o nome da tabela e a função associada
    opcoes = [
        ("DF0: Tabela Saúde Mental", partial(tabela, df_0)),
        ("DF1: Depressão em Homens e Mulheres (%)", partial(tabela, df_1)),
        ("DF2: Suicídio e Depressão na População", partial(tabela, df_2)),
        ("DF3: Depressão na População", partial(tabela, df_3)),
        ("Estatísticas: Tabela Saúde Mental", partial(tabela_describe, df_0)),
        ("Estatísticas: Depressão em Homens e Mulheres (%)", partial(tabela_describe, df_1)),
        ("Estatísticas: Suicídio e Depressão na População", partial(tabela_describe, df_2)),
        ("Estatísticas: Depressão na População", partial(tabela_describe, df_3))
    ]

    # Chama a função do menu e passa o título 'Tabelas' e a lista de opções
    return menu('Tabelas', opcoes)


# %% [markdown]
# ##### 2.4.4. Função principal

# %%
def main():
    # Lista de opções para o menu principal, cada opção é um tuplo contendo o nome da categoria e a função associada
    opcoes = [
        ("Gráficos", graficos),
        ("Tabelas", tabelas),
    ]

    # Chama a função do menu principal e passa o título 'Tendências globais em matéria de saúde mental' e a lista de opções
    return menu('Tendências globais em matéria de saúde mental', opcoes)

# Inicia a execução do programa chamando a função main()
main()


# %% [markdown]
# ### 3. Conclusão 


