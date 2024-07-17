package com.spring.biz.board.impl;

import java.util.List;

import com.spring.biz.board.BoardVO;

public interface BoardService {
	
	public void insertBoard(BoardVO vo);
	public void updateBoard(BoardVO vo);
	public void deleteBoard(BoardVO vo);
	public BoardVO getBoard(BoardVO vo);
	public List<BoardVO> getBoardList();
}
