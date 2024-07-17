package com.example.spring42.service;

import com.example.spring42.domain.Article;
import com.example.spring42.dto.UpdateArticleRequest;
import com.example.spring42.repository.BlogRepository;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import javax.transaction.Transactional;
import java.util.List;

@Service
public class BlogService {

    @Autowired
    BlogRepository blogRepository;
    public List<Article> findAll(){
        return blogRepository.findAll();
    }

    public Article findById(Long id){
        return blogRepository.findById(id).orElseThrow(()->new IllegalArgumentException(id+"가 없습니다."));
    }

    public void Delete(Long id){
        blogRepository.deleteById(id);
    }

    @Transactional
    public Article update(long id, UpdateArticleRequest request){
        Article article = blogRepository.findById(id).orElseThrow(()->new IllegalArgumentException("찾을 수 없습니다."));
        article.update(request.getTitle(),request.getContent());
        return article;
    }

    public void insert(Article article){
    }
}
