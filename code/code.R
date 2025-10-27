# ===================================================================
# 1. PACOTES
# ===================================================================
library(dplyr)
library(ggplot2)
library(ggstatsplot)
library(ggthemes)
library(ggridges)
library(readr)
library(stringr)
library(tidyr)
library(forcats)
library(scales)
library(viridis)

# ===================================================================
# 2. CAMINHOS E DADOS
# ===================================================================
data_file <- "C:/Users/PC GAMER/Downloads/erp-ofdi-model/data/data.csv"
output_dir <- "C:/Users/PC GAMER/Downloads/erp-ofdi-model/results/R"

if (!dir.exists(output_dir)) {
  dir.create(output_dir, recursive = TRUE)
}

dados_base <- readr::read_csv(data_file, col_types = cols(.default = "c"))
names(dados_base) <- gsub("\\s+", ".", names(dados_base))
dados <- dados_base

# ===================================================================
# 3. LIMPEZA E PREPARAÇÃO
# ===================================================================

# --- 3.1. Limpar "Quantity.in.Millions" ---
parse_num <- function(x) {
  x_clean <- as.character(x)
  x_clean <- gsub("[^0-9\\.,-]", "", x_clean)
  x_clean <- gsub(",", "", x_clean)
  suppressWarnings(as.numeric(x_clean))
}

dados$Quantity.in.Millions <- parse_num(dados$Quantity.in.Millions)
dados$Year <- as.numeric(dados$Year)

dados_clean <- dados %>%
  filter(!is.na(Quantity.in.Millions), !is.na(Year))

# --- 3.2. Criar Variáveis "Going Global" ---
dados_clean <- dados_clean %>%
  mutate(
    Going_Global1.0 = as.numeric(Year %in% c(2005:2011)),
    Going_Global2.0 = as.numeric(Year %in% c(2012:2016)),
    Going_Global3.0 = as.numeric(Year %in% c(2017:2024)),
    Change_Startegy = as.numeric(Year %in% c(2016:2024))
  )

# --- 3.3. Criar Variáveis Categóricas (BRI, Greenfield) ---
dados_clean$bri_grp <- ifelse(grepl("^(1|bri|y|yes|sim|true)$", tolower(trimws(dados_clean$BRI))), "BRI", "Others")
dados_clean$GF <- ifelse(grepl("^(g|greenfield|1|y|yes|sim|true)$", tolower(trimws(dados_clean$Greenfield))), "Greenfield", "Other")

# --- 3.4. Limpar e Binar "Share.Size" ---
dados_clean$Share.Size.Numeric <- parse_num(dados_clean$Share.Size)
dados_clean <- dados_clean %>%
  mutate(
    Share.Bin = cut(Share.Size.Numeric,
                    breaks = c(-Inf, 19, 39, 59, 79, Inf),
                    labels = c("0-19", "20-39", "40-59", "60-79", "80-99"),
                    right = TRUE, include.lowest = TRUE
    )
  )

# --- 3.5. Preparar Fatores ---
cols_to_factor <- c("Subsector", "Region", "Month", "Share.Size", "Transaction.Party", "Sector", "bri_grp", "GF", "Year")
dados_clean[cols_to_factor] <- lapply(dados_clean[cols_to_factor], function(x) {
  if (is.character(x)) {
    forcats::fct_na_value_to_level(as.factor(x), "Unknown")
  } else {
    as.factor(x)
  }
})

month_levels <- c("January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December", "Unknown")
dados_clean$Month <- factor(dados_clean$Month, levels = intersect(month_levels, levels(dados_clean$Month)))

# --- 3.6. Filtrar Outliers (para visualização) ---
dados_filtered <- dados_clean %>%
  filter(Quantity.in.Millions > 0, Quantity.in.Millions <= 1200)

# ===================================================================
# 4. GRÁFICOS DE DENSIDADE (Joy Plots)
# ===================================================================

create_density_plot <- function(data, y_col_name, y_label, file_name, scale = 3) {
  p <- ggplot(data, aes(x = Quantity.in.Millions, y = .data[[y_col_name]], fill = ..x..)) +
    ggridges::geom_density_ridges_gradient(scale = scale, rel_min_height = 0.01) +
    scale_fill_viridis_c(option = "plasma", name = "Amount (Millions)") +
    labs(
      title = "Amount of Investment - in Millions",
      x = "Quantity in Millions",
      y = y_label
    ) +
    theme_minimal() +
    theme(legend.position = "none")
  
  ggsave(file.path(output_dir, file_name), p, width = 10, height = 8, dpi = 150)
  return(p)
}

create_density_plot(dados_filtered, "Subsector", "Subsector", "density_subsector.png")
create_density_plot(dados_filtered, "Region", "Region", "density_region.png")
create_density_plot(dados_filtered, "Month", "Month", "density_month.png")
create_density_plot(dados_filtered, "Sector", "Sector", "density_sector.png")
create_density_plot(dados_filtered, "bri_grp", "BRI", "density_bri.png", scale = 2)

top_shares <- dados_filtered %>% count(Share.Size, sort = TRUE) %>% top_n(30) %>% pull(Share.Size)
data_share <- dados_filtered %>% filter(Share.Size %in% top_shares)
create_density_plot(data_share, "Share.Size", "Share Size", "density_share_size.png")

top_parties <- dados_filtered %>% count(Transaction.Party, sort = TRUE) %>% top_n(25) %>% pull(Transaction.Party)
data_party <- dados_filtered %>% filter(Transaction.Party %in% top_parties)
create_density_plot(data_party, "Transaction.Party", "Transaction Party", "density_transaction_party.png")


# ===================================================================
# 5. GRÁFICOS DE BARRAS E HISTOGRAMAS
# ===================================================================

# --- 5.1. Gráfico ---
df_year_counts <- dados_clean %>%
  count(Year) %>%
  mutate(Proportion = n / sum(n))

p_bar_year <- ggplot(df_year_counts, aes(x = Year, y = n)) +
  geom_bar(stat = "identity", fill = "#0c4c8a", alpha = 0.8) +
  geom_text(
    aes(label = scales::percent(Proportion, accuracy = 0.1)),
    vjust = -0.5, size = 3
  ) +
  labs(
    title = "Investment Count by Year",
    y = "Count",
    x = "Year"
  ) +
  ggthemes::theme_tufte() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

ggsave(file.path(output_dir, "bar_count_year.png"), p_bar_year, width = 12, height = 8, dpi = 150)


# --- 5.2. Histograma: "Going Global Phase" ---
df_gg_hist <- dados_filtered %>%
  select(Going_Global1.0, Going_Global2.0, Going_Global3.0, Quantity.in.Millions) %>%
  pivot_longer(
    cols = starts_with("Going_Global"),
    names_to = "Going_Global_Phase",
    values_to = "is_in_phase"
  ) %>%
  filter(is_in_phase == 1) %>%
  mutate(
    Going_Global_Phase = gsub("Going_Global", "Going Global ", Going_Global_Phase),
    Going_Global_Phase = gsub("\\.0", "", Going_Global_Phase),
    Going_Global_Phase = trimws(Going_Global_Phase)
  )

p_hist_gg <- grouped_gghistostats(
  data = df_gg_hist,
  x = Quantity.in.Millions,
  grouping.var = Going_Global_Phase,
  type = "p",
  ggtheme = ggthemes::theme_tufte(),
  messages = FALSE
  
) + 
  labs(
    title = NULL,
    x = "Investment Amount (Millions)",
    y = "Frequency"
  ) +
  theme(
    plot.title = element_blank(),
    plot.margin = margin(t = 20, r = 10, b = 10, l = 10, unit = "pt"),
    strip.text = element_text(
      face = "bold", 
      size = 12,
      hjust = 0.5,
      margin = margin(t = 15, b = 15)
    ),
    plot.subtitle = element_text(margin = margin(b = 15))
  )


p_hist_gg$patches$plots <- lapply(p_hist_gg$patches$plots, function(plot) {
  plot + theme(
    plot.margin = margin(t = 15, r = 5, b = 15, l = 5, unit = "pt"),  
    plot.title = element_text(
      hjust = 0.5, 
      face = "bold",
      size = 12,
      margin = margin(b = 10)  
    )
  )
})

# Garantir que todos os títulos estejam visíveis
p_hist_gg$patches$plots <- p_hist_gg$patches$plots

ggsave(
  file.path(output_dir, "histogram_by_going_global.png"), 
  p_hist_gg, 
  width = 20,
  height = 8,  
  dpi = 150
)

# --- 5.3. Gráfico de Barras: Going Global Share ---
df_gg_summary <- dados_clean %>%
  filter(!is.na(Share.Bin)) %>%
  select(Going_Global1.0, Going_Global2.0, Going_Global3.0, Quantity.in.Millions, Share.Bin) %>%
  pivot_longer(
    cols = starts_with("Going_Global"),
    names_to = "Going_Global_Phase",
    values_to = "is_in_phase"
  ) %>%
  filter(is_in_phase == 1) %>%
  group_by(Going_Global_Phase, Share.Bin) %>%
  summarise(Amount_investment = sum(Quantity.in.Millions, na.rm = TRUE), .groups = 'drop') %>%
  group_by(Going_Global_Phase) %>%
  mutate(Proportion = Amount_investment / sum(Amount_investment)) %>%
  ungroup() %>%
  mutate(
    Going_Global_Phase = gsub("_", " ", Going_Global_Phase),
    Going_Global_Phase = gsub("\\.0", ".0", Going_Global_Phase)
  )

p_gg_share <- ggplot(df_gg_summary, aes(x = Share.Bin, y = Amount_investment)) +
  geom_bar(stat = "identity", fill = "darkgrey") +
  geom_text(
    aes(label = scales::percent(Proportion, accuracy = 0.1)),
    vjust = -0.5, size = 3
  ) +
  facet_wrap(~Going_Global_Phase, scales = "free_y") +
  scale_y_continuous(labels = scales::comma) +
  labs(
    y = "Amount of Investment",
    x = "Market Share (%)"
  ) +
  theme_minimal(base_size = 12) +
  theme(
    strip.text = element_text(color = "red", face = "bold", size = 14),
    panel.grid.major.x = element_blank(),
    panel.grid.minor.y = element_blank()
  )

ggsave(file.path(output_dir, "going_global_share.png"), p_gg_share, width = 14, height = 7, dpi = 150)


# ===================================================================
# 6. GRÁFICOS ADICIONAIS
# ===================================================================

# --- 6.1. Boxplot: Quantidade por Ano ---
p_boxplot_year <- ggbetweenstats(
  data = dados_clean,
  x = Year,
  y = Quantity.in.Millions,
  xlab = "Year",
  ylab = "Quantity in Millions",
  type = "p",
  pairwise.display = "s",
  ggtheme = ggthemes::theme_tufte(),
  package = "ggsci",
  palette = "default_ucscgb",
  outlier.tagging = TRUE
)
ggsave(file.path(output_dir, "boxplot_year.png"), p_boxplot_year, width = 14, height = 8, dpi = 150)

# --- 6.2. Gráfico de Rosca (Doughnut) por Ano ---
invest_year_sum <- dados_clean %>%
  group_by(Year) %>%
  summarise(Amount_investment = sum(Quantity.in.Millions, na.rm = TRUE))

p_doughnut_year <- ggplot(invest_year_sum, aes(x = 2, y = Amount_investment, fill = as.factor(Year))) +
  geom_bar(stat = "identity", width = 1) +
  coord_polar(theta = "y", start = 0) +
  xlim(0.5, 2.5) +
  theme_void() +
  theme(legend.position = "none") +
  geom_text(aes(label = Year), position = position_stack(vjust = 0.5)) +
  labs(title = "Investment Amount by Year")

ggsave(file.path(output_dir, "doughnut_year.png"), p_doughnut_year, width = 8, height = 8, dpi = 150)


# --- 6.3. Boxplots: Greenfield e BRI ---
df_gf <- dados_clean %>%
  filter(!is.na(GF), Quantity.in.Millions <= 1100) %>%
  mutate(.ylabel = round(Quantity.in.Millions, 0))

p_gf <- ggbetweenstats(
  data = df_gf,
  x = GF,
  y = Quantity.in.Millions,
  type = "p",
  ggtheme = ggthemes::theme_tufte(),
  pairwise.display = "s",
  outlier.tagging = TRUE,
  outlier.label = .ylabel,
  messages = FALSE
)
ggsave(file.path(output_dir, "boxplot_greenfield.png"), p_gf, width = 11, height = 7, dpi = 150)

df_bri <- dados_clean %>%
  filter(!is.na(bri_grp), Quantity.in.Millions <= 1100) %>%
  mutate(.ylabel = round(Quantity.in.Millions, 0))

p_bri <- ggbetweenstats(
  data = df_bri,
  x = bri_grp,
  y = Quantity.in.Millions,
  type = "p",
  ggtheme = ggthemes::theme_tufte(),
  pairwise.display = "s",
  outlier.tagging = TRUE,
  outlier.label = .ylabel,
  messages = FALSE
)
ggsave(file.path(output_dir, "boxplot_bri.png"), p_bri, width = 11, height = 7, dpi = 150)

print(paste("Todos os gráficos foram salvos em:", output_dir))

