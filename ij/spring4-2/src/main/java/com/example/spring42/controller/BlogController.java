package com.example.spring42.controller;

import com.example.spring42.domain.Article;
import com.example.spring42.dto.ArticleListViewResponse;
import com.example.spring42.dto.ArticleViewResponse;
import com.example.spring42.dto.UpdateArticleRequest;
import com.example.spring42.service.BlogService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.ResponseEntity;
import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.*;

import java.util.ArrayList;
import java.util.List;

@Controller
public class BlogController {
    @Autowired
    private BlogService blogService;

    @GetMapping("/articles")
    public String getArticles(Model model) {
        List<ArticleListViewResponse> articles = new ArrayList<>() ;
        for(Article article : blogService.findAll()){
            articles.add(new ArticleListViewResponse(article));
        }

        model.addAttribute( "articles",articles);
        return "articleList";
    }

    @GetMapping("/articles/{id}")
    public String getArticle(@PathVariable Long id, Model model) {
        Article article = blogService.findById(id);
        model.addAttribute("article",new ArticleViewResponse(article));
        return "article";
    }

    @DeleteMapping("/api/article/{id}")
    public String deleteArticle(@PathVariable Long id, Model model) {
        blogService.Delete(id);
        return "redirect:/articles";
    }

    @GetMapping("/new-article")
    public String newArticle(@RequestParam(required = false) Long id, Model model) {
        //신규
        if (id == null) {
            return "redirect:/articles";
        }
        //수정
        else{
            Article article = blogService.findById(id);
            model.addAttribute("article",new ArticleViewResponse(article));
            return "newArticle";
        }
    }

    @PutMapping("/api/article/{id}")
    public ResponseEntity<Article> updateArticle(@PathVariable Long id, @RequestBody UpdateArticleRequest request) {
        Article updatedArticle = blogService.update(id, request);
        return ResponseEntity.ok().body(updatedArticle);
    }

}
