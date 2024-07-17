package com.spring.biz.board.impl;

import java.util.List;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Qualifier;
import org.springframework.stereotype.Service;

import com.spring.biz.board.BoardVO;

@Service("boardService")
public class BoardServiceImpl implements BoardService{
	
	@Autowired
	@Qualifier("boardDAOSpring")
	private BoardDAOSpring boardDAO;
	
	//생성자로 구현
	/*	
	public BoardServiceImpl(BoardDAO boardDAO) {
		this.boardDAO = boardDAO;
	}
	*/
	//set메서드로 구현
	/*
	public void setBoardDAO(BoardDAO boardDAO) {
		this.boardDAO = boardDAO;
	}
	*/

	@Override
	public void insertBoard(BoardVO vo) {
		System.out.println("[insertBoard]메서드 실행");
		
		boardDAO.insertBoard(vo);
	}

	@Override
	public void updateBoard(BoardVO vo) {
		System.out.println("[updateBoard]메서드 실행");
		boardDAO.updateBoard(vo);
		
	}

	@Override
	public void deleteBoard(BoardVO vo) {
		System.out.println("[deleteBoard]메서드 실행");
		boardDAO.deleteBoard(vo);
		
	}

	@Override
	public BoardVO getBoard(BoardVO vo) {
		System.out.println("[getBoard]메서드 실행");
		return boardDAO.getBoard(vo);
	}

	@Override
	public List<BoardVO> getBoardList(BoardVO vo) {
		System.out.println("[getBoardList]메서드 실행");
		return boardDAO.getBoardList(vo);
	}

}
