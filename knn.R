library(shiny)
ui <- fluidPage(
  titlePanel("Определение модели самолёта"),
  
  sidebarLayout(
    sidebarPanel(),
    
    mainPanel(
      plotOutput(outputId = "tmp_plot")
    )
  )
)